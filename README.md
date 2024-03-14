LabelAnything
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── checkpoints        <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── label_anything     <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts of the models
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

* HOW TO CREATE A VENV FOR CINECA
- we need to add another file for cineca requirements
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_cineca.txt
```

* GENERAL PURPOSE ENVIRONMENT
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


* DOWNLOAD DRAM DATASET
```
wget -P  ./data/raw https://faculty.runi.ac.il/arik/site/artseg/DRAM_processed.zip
unzip ./data/raw/DRAM_processed.zip -d ./data/raw
unrar x ./data/raw/DRAM_processed.rar ./data/raw
```


* PROMPT ENCODER PRETRAINING
To train the prompt encoder on CINECA, you can run the command 
```
sbatch pretrain_pe_parallel
```
Then, once the pre-training phase is completed, move the checkpoint from the out directory to chrckpoint running:
```
cp path/to/out/dir/pytorch_model_1.bin path/to/checkpoint/dir
cd path/to/checkpoint/dir/
mv pytorch_model_1.bin model.bin
```
Finally, you can train LabelAnything model by running the command:
```
python3 experiment --parameters="parameters/COCO_vit.yaml"
```
If you extracted the embeddings you can run the command:
```
python3 experiment --parameters="parameters/COCO.yaml"
```
If you have a multi GPU machine, you can run the command:
``` 
accelerate launch --multi_gpu main.py experiment --parameters="parameters/COCO.yaml"
accelerate launch --multi_gpu main.py experiment --parameters="parameters/COCO_vit.yaml"  
```

#### Set up the pretrained weights

You can download the pretrained weights from the following links:
Fold 0: https://drive.google.com/file/d/1aTyBM1geU28tZU-KhIEV8k5FBGZUQuRD/view?usp=drive_link
Fold 1: https://drive.google.com/file/d/1UbVGSxSWVC947g8-GCGVf9btvLzNh2SA/view?usp=drive_link
Fold 2: https://drive.google.com/file/d/12d-dPq5Wjz5YakymTEefnzC__1YTq9ue/view?usp=drive_link
Fold 3: https://drive.google.com/file/d/1EVYhVP3H82LN7HP3zMDFskchbGfWND0H/view?usp=drive_link

Create the folders for the weights
```bash
# Fold 0
mkdir -p wandb/offline-run-20240306_195048-n9623ye5/files/best
# Fold 1
mkdir -p wandb/offline-run-20240306_214217-3ndl7had/files/best
# Fold 2
mkdir -p wandb/offline-run-20240306_214216-b3wov30u/files/best
# Fold 3
mkdir -p wandb/offline-run-20240306_214216-cwrjbktc/files/best
```
Move each weight to the corresponding folder

```bash
# Fold 0
mv pytorch_model_0.bin wandb/offline-run-20240306_195048-n9623ye5/files/best/pytorch_model.bin
# Fold 1
mv pytorch_model_1.bin wandb/offline-run-20240306_214217-3ndl7had/files/best/pytorch_model.bin
# Fold 2
mv pytorch_model_2.bin wandb/offline-run-20240306_214216-b3wov30u/files/best/pytorch_model.bin
# Fold 3
mv pytorch_model_3.bin wandb/offline-run-20240306_214216-cwrjbktc/files/best/pytorch_model.bin
```

Run the validations
```bash
# Fold 0
python3 validate.py --parameters="parameters_validation/COCO_Fold0.yaml"
# Fold 1
python3 validate.py --parameters="parameters_validation/COCO_Fold1.yaml"
# Fold 2
python3 validate.py --parameters="parameters_validation/COCO_Fold2.yaml"
# Fold 3
python3 validate.py --parameters="parameters_validation/COCO_Fold3.yaml"
``` 