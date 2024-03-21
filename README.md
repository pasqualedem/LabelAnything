# Label Anything

This repository contains the official code for the paper "LabelAnything: Multi-Class Few-Shot Semantic Segmentation with Visual Prompts".

![Label Anything](assets/la.png)

## Requirements

**Note**: The following instructions are for a Linux environment using CUDA 12.1. 

Create a virtual environment using our conda environment file:

```bash
conda env create -f label-anything.yml
conda activate label_anything
```

## Prepare the Datasets

Enter the `data` directory, create and enter the directory `coco` and download the COCO 2017 train and val images and the COCO 2014 annotations from the [COCO website](https://cocodataset.org/#download):

```bash
cd data
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

Unzip the files:

```bash
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2014.zip
rm -rf train2017.zip val2017.zip annotations_trainval2014.zip
```

The `coco` directory should now contain the following files and directories:

```
coco
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
|   ├── person_keypoints_train2014.json
|   └── person_keypoints_val2014.json
├── train2017
└── val2017
```

Now, join the images of the train and val sets into a single directory:

```bash
mv val2017/* train2017
mv train2017 train_val_2017
rm -rf val2017
```

Finally, you will have to rename image filenames in the COCO 2014 annotations to match the filenames in the `train_val_2017` directory. To do this, run the following script:

```bash
python main.py rename_coco20i_json --instances-path data/coco/annotations/instances_train2014.json
python main.py rename_coco20i_json --instances-path data/coco/annotations/instances_val2014.json
```

## Preprocess

We use [Segment Anything](https://github.com/facebookresearch/segment-anything) pretrained models to extract image features. Enter the `checkpoints` directory and download the pretrained models from the Segment Anything repository:

```bash
mkdir offline
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**Optional**: To optimize model training and evaluation, you can extract the output of the vision encoder for each image in the COCO dataset, and save it to disk. We call `last_hidden_state` the directory containing the output of the convolutional neck we added on top of the Vision Transformer, while we call `last_block_state` the final output of ViT. This can be done by running the following script:

```bash
mkdir -p data/coco/vit_sam_embeddings/last_hidden_state
mkdir data/coco/vit_sam_embeddings/last_block_state
python main.py preprocess --encoder vit_b --checkpoint checkpoints/sam_vit_b_01ec64.pth --use_sam_checkpoint --directory data/coco/train_val_2017 --batch_size 16 --num_workers=8 --outfolder data/coco/vit_sam_embeddings/last_hidden_state --last_block_dir data/coco/vit_sam_embeddings/last_block_state
```

## Train and Test

You can train LabelAnything model on COCO-20i by running the command:

```bash
python main.py experiment --parameters="parameters/COCO_vit.yaml"
```

If you extracted the embeddings you can run the command:

```bash
python main.py experiment --parameters="parameters/COCO.yaml"
```

By default, four training processes will be launched sequentially, one for each fold of the 4-fold cross-validation. It is possible to launch only interesting training by deleting them from the `other_grids` section of the parameter file. Remember to also change the `val_fold_idx` in the `parameters.dataset` section to the fold you want to validate, which will be executed at the beginning. If you start a model training, you don't need to run the the validation step, as it is already included in the training process.

If you have a multi GPU machine, you can run the command:

```bash
accelerate launch --multi_gpu main.py experiment --parameters="parameters/COCO.yaml"
accelerate launch --multi_gpu main.py experiment --parameters="parameters/COCO_vit.yaml"  
```

Experiments are tracked using [Weights & Biases](https://wandb.ai/site). The resulting run files are stored in the `offline/wandb/run-<date>-<run_id>` directory. Model weights for the specific run are saved in the `files` subdirectory of the run folder.


## Test

To protect anonimity, our pretrained models are not available for download. Model weights will be available upon acceptance.

## Demo

If you have trained the model and want to use it in an interactive way to segment images, you can run the following command:

```bash
python -m streamlit run app.py
```

In the web interface, enter the Weights & Biases path to a run id `<entity>/<project>/<run-id>` ([help](https://docs.wandb.ai/ref/python/public-api/api#run)) of the model you want to use. Currently, the demo only supports box annotations. You will be asked to enter a query image, class names, and support images with prompts.

## Project Organization

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

## Old Stuff

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