import torch
from torch import nn
from torchvision.models import resnet
from torchvision import transforms
from typing import Tuple, Callable
from torchvision.transforms import v2
import timm
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class VirchowWrapper(torch.nn.Module):
    def __init__(self, virchow2_model: torch.nn.Module):
        super().__init__()
        self.model = virchow2_model

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)  # size: B x 261 x 1280

        class_token = output[:, 0]  # size: B x 1280
        patch_tokens = output[:, 5:]  # size: B x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: B x 2560

        return embedding


def from_name(name: str) -> Tuple[nn.Module, int, Callable]:
    """
    :param name: Model name (uni, virchow2, resnet50, resnet18, kaiko-*)
    :return: (image encoder, encoding dimension, preprocessing transform)
    """
    name = name.lower()
    if name == "uni":
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        return model.eval(), 1024, transform

    elif name == "virchow2":
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = model.eval()
        preprocessing = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        wrapped_model = VirchowWrapper(model).eval()
        return wrapped_model, 2560, preprocessing

    elif name.startswith("kaiko"):
        preprocessing = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=224),
                v2.CenterCrop(size=224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )
        model_name = name.split("-")[1]
        assert model_name in ["vits16", "vits8", "vitb16", "vitb8", "vitl14"], f"Unknown Kaiko-ai ViT: {model_name}"
        if model_name.startswith("vits"):
            dim = 384
        elif model_name.startswith("vitb"):
            dim = 768
        else:
            dim = 1024
        model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", model_name, trust_repo=True).eval()
        return model, dim, preprocessing

    elif name.startswith("resnet"):
        if name == "resnet50":
            model = resnet.resnet50(resnet.ResNet50_Weights.IMAGENET1K_V1)
        elif name == "resnet18":
            model = resnet.resnet18(resnet.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise NotImplementedError(f"Unknown resnet type: {name}")
        dim = model.fc.in_features
        model.fc = nn.Identity()  # do not perform classification layer
        return model, dim, nn.Identity()

    else:
        raise ValueError(f"Invalid patch encoder '{name}'.")