# PATHS: A Hierarchical Transformer for Efficient Whole Slide Image Analysis

![](assets/paths_overall.png)


## Usage
### Installation
First, create a virtual env
```
git clone <repository_path>
cd PATHS
conda env create --name paths --file=environment.yml
```
note: we recommend `mamba` over `conda` for faster installation.

### Data Download
Download from the TCGA data portal using the GDC manifests provided in `data/gdc_manifests`. 
More details to follow in camera-ready version.

### Data Preprocessing
Replace `/PATH/TO/SLIDES` with the root of the dataset containing the TCGA WSIs, and `PATH/TO/OUT/DIR` with the
directory in which preprocessed features should be stored. Then run the following command (with modification):
```
python preprocess/preprocess.py --model UNI --dir /PATH/TO/SLIDES --out PATH/TO/OUT/DIR --batch 32 --patch 256 --workers 32 --magnifications 0.625 1.25 2.5 5.0 10.0 --downscale 4
```
Full explanation of arguments may be found in preprocess.py.

### Training
First, create a new directory in `models` (or elsewhere) which contains a `config.json` file.
The config specifies all aspects of the model and training. The provided sample config `models/sample/config.json`
may be of use. We additionally provide a pre-trained model and config in `models/brca_paths_0`.

Then, replacing `CONFIG_DIRECTORY`, run
```
python train.py -m models/CONFIG_DIRECTORY
```
The model will be saved to the config's directory.

### Visualisation
Run the below command to create a heatmap visualisation from any local WSI.
Replace `/PATH/TO/SLIDE` and `CONFIG_DIRECTORY` with appropriate values.
If the slide is a CAMELYON17 slide, set `/PATH/TO/XML` to the path to the annotation XML.
If not, remove the `--annotation-path` argument.
```
python heatmap_visualise.py -m models/CONFIG_DIRECTORY --slide-path /PATH/TO/SLIDE --annotation-path /PATH/TO/XML --out heatmap.pdf
```
Note that we provide a single pre-trained model in the directory `brca_paths_0` - even without the datasets downloaded,
it should be possible to run visualisation on any WSI of your choice. However, note that this model was trained on
TCGA-BRCA, and application to other tissue types may produce inaccurate results.
