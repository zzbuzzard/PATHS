import multiprocessing
import torch
import argparse
import utils
import os
from os.path import join
import torchvision.transforms.functional as trf
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
from tqdm import tqdm
import logging
from multiprocessing import Pool
import traceback

from model.image_encoder import from_name

LM = 2
print("LOAD MODE", LM)


@torch.no_grad()
def process(args, model, model_dim, transform, wsi: WSIReader, power: float, tissue_threshold: float = 0.1, load_mode=LM,
            out_device="cpu", progress_bar=False):
    ht, wt = wsi.slide_dimensions(resolution=power, units="power")
    ht, wt = utils.next_multiple(ht, args.patch), utils.next_multiple(wt, args.patch)

    masker = OtsuTissueMasker()
    bp = power / args.downscale
    hb, wb = ht // args.downscale, wt // args.downscale
    img = wsi.read_rect((0, 0), (hb, wb), resolution=bp, units="power")
    mask = masker.fit_transform([img])[0]

    def get_proportion(h, w):
        h, w = h // args.downscale, w // args.downscale
        p = args.patch // args.downscale
        s = mask[w: w + p, h: h + p]
        return s.sum() / s.size

    def get_img(h, w):
        if load_mode == 1:
            return massive_image[w: w + args.patch, h: h + args.patch]
        elif load_mode == 2:
            return wsi.read_rect((h, w), (args.patch, args.patch), resolution=power, units="power", coord_space="resolution")

    print(power,":",ht,wt)

    if load_mode == 1:
        # up to 22GB ...
        massive_image = wsi.read_rect((0, 0), (ht, wt), resolution=power, units="power")

    h_patches = ht // args.patch
    w_patches = wt // args.patch
    out = torch.zeros((w_patches, h_patches, model_dim)).to(out_device)

    hws = []
    for h in range(0, ht, args.patch):
        for w in range(0, wt, args.patch):
            if get_proportion(h, w) > tissue_threshold:
                hws.append((h, w))

    print("Num patches:", len(hws), "/", (w_patches * h_patches))
    it = range(0, len(hws), args.batch)
    if progress_bar:
        it = tqdm(it)
    for s in it:
        e = min(s + args.batch, len(hws))
        imgs = [trf.to_tensor(get_img(h, w))[None] for h, w in hws[s: e]]
        imgs = torch.cat(imgs, dim=0)
        imgs = imgs.to(utils.device)
        
        imgs = transform(imgs)

        # (B x C x H x W) -> (B x D)
        emb = model(imgs)

        for i, (h, w) in enumerate(hws[s: e]):
            out[w // args.patch, h // args.patch] = emb[i]

    return out


def process_slide(t):
    slide, model, model_dim, transform, args = t
    wsi = WSIReader.open(join(args.dir, slide))
    if wsi.info.objective_power is None:
        print("No objective power; assuming 40")
        wsi._m_info.objective_power = 40

    slide_id = ".".join(slide.split(".")[:-1])  # remove .svs

    if len(wsi.info.slide_dimensions) == 1:
        print("Warning: slide", slide_id, "has slide_dimensions", wsi.info.slide_dimensions)

    for p in args.magnifications:
        save_path = join(args.out, f"{slide_id}_{p:.3f}.pt")

        if os.path.isfile(save_path):
            continue

        try:
            out = process(args, model, model_dim, transform, wsi, p)
            torch.save(out, save_path)
        except:
            print("ISSUE WITH SLIDE", slide, "AT POWER", p)
            traceback.print_exc()

    wsi.openslide_wsi.close()


logger = logging.getLogger()
logger.setLevel(logging.ERROR)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=["resnet50", "resnet18", "UNI"],
                        help="Patch processing model", default="resnet50")
    parser.add_argument("-d", "--dir", type=str, help="Path to input data folder")
    parser.add_argument("-o", "--out", type=str, help="Path to output data folder")
    parser.add_argument("-b", "--batch", type=int, help="Batch size")
    parser.add_argument("-p", "--patch", type=int, help="Patch size", default=256)
    parser.add_argument("-w", "--workers", type=int, help="Number of parallel processes to run", default=32)
    parser.add_argument("-ms", "--magnifications", type=float, nargs="+", help="Magnification levels",
                        default=[0.625, 1.25, 2.5, 5.0, 10.0])
    parser.add_argument("-ds", "--downscale", type=int, help="Magnification downscale amount for background masking. "
                                                             "E.g. downscale=4, the image is processed at 4x less "
                                                             "magnification before being passed to the background "
                                                             "masker.", default=4)

    # Naming convention:
    #  [slide ID]_m = [H x W x D]
    #   entirely 0 entry means background
    #  where m = magnification power

    args = parser.parse_args()

    if not os.path.exists(args.out):
        print("Creating directory", args.out)
        os.makedirs(args.out)

    model, *x = from_name(args.model)
    if len(x) == 1:
        import torchvision.transforms as tr
        model_dim = x[0]
        transform = tr.Compose([])
    else:
        model_dim, transform = x
    model = model.eval().to(utils.device)

    slide_ids = [i for i in os.listdir(args.dir) if i.endswith(".svs")]

    # KIRP slides with only one level; loading at 0.625 etc extremely slow
    # slide_ids.remove("TCGA-UZ-A9PQ-01Z-00-DX1.C2CB0E94-2548-4399-BCAB-E4D556D533EF.svs")
    # slide_ids.remove("TCGA-5P-A9KC-01Z-00-DX1.F3D67C35-111C-4EE6-A5F7-05CF8D01E783.svs")
    # slide_ids.remove("TCGA-5P-A9KA-01Z-00-DX1.6F4914E0-AB5D-4D5F-8BF6-FB862AA63A87.svs")

    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    data = [(s, model, model_dim, transform, args) for s in slide_ids]

    with Pool(processes=args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_slide, data), total=len(slide_ids)):
            pass
