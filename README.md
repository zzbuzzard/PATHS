# PATHS: A Hierarchical Transformer for Efficient Whole Slide Image Analysis




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

### Data Preprocessing
Replace `[DS_PATH]` with the root of the dataset containing the TCGA WSIs, and `[OUT]` with the directory in which
preprocessed features should be stored. Then run the following command (with modification):
```
python preprocess/preprocess.py --model UNI --dir [DS_PATH] --out [OUT] --batch 32 --patch 256 --workers 32 --magnifications 0.625 1.25 2.5 5.0 10.0 --downscale 4
```
Full explanation of arguments may be found in preprocess.py.

### Training
First, create a new directory in `models` (or elsewhere) which contains a `config.json` file.
The config specifies all aspects of the model and training. The provided sample config `models/sample/config.json`
will be of use.

Then, run
```
python train.py -m models/[config_directory]
```
The model will be saved to the config's directory.

### Visualisation
TODO