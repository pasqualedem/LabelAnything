[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/label-anything-multi-class-few-shot-semantic/few-shot-semantic-segmentation-on-coco-20i-2-1)](https://paperswithcode.com/sota/few-shot-semantic-segmentation-on-coco-20i-2-1?p=label-anything-multi-class-few-shot-semantic)

# [Label Anything](https://arxiv.org/abs/2407.02075)

This repository contains the official code for the paper ["LabelAnything: Multi-Class Few-Shot Semantic Segmentation with Visual Prompts"](https://arxiv.org/abs/2407.02075) accepted at [ECAI 2025](https://ecai2025.org/).

![Label Anything](assets/la.png)

## Demo

Easily run the demo through [uv](https://docs.astral.sh/uv/) by executing the following command:

```bash
uvx --from git+https://github.com/pasqualedem/LabelAnything app
```

## Installation

Or you can install the package manually by cloning the repository and installing the dependencies:
**Note**: The following instructions are for a Linux environment using CUDA 12.1. 

Create a virtual environment using uv

```bash
uv sync
source .venv/bin/activate
```

## Released checkpoints

| Encoder | Embedding Size | Image Size | Fold |Checkpoint |
|---------|----------------|------------|------|------------|
| SAM     | 512            | 1024       | -    | [![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-000000?style=flat-square&logo=huggingface)](https://huggingface.co/pasqualedem/label_anything_sam_1024_coco)
| ViT-MAE | 256            | 480        | -    | [![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-000000?style=flat-square&logo=huggingface)](https://huggingface.co/pasqualedem/label_anything_mae_480_coco) |
| ViT-MAE | 256            | 480        | 0    | [![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-000000?style=flat-square&logo=huggingface)](https://huggingface.co/pasqualedem/label_anything_coco_fold0_mae_7a5p0t63) |

Import them with the following command:

```python
from label_anything.models import LabelAnything
model = LabelAnything.from_pretrained("pasqualedem/label_anything_sam_1024_coco")
```

## Training

You need to download the COCO 2017 dataset to train the model. The following sections describe how to set up these datasets.

### Setting up [COCO 2017](https://cocodataset.org/#home) Dataset with COCO 2014 annotations

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
python main.py rename_coco20i_json --instances_path data/coco/annotations/instances_train2014.json
python main.py rename_coco20i_json --instances_path data/coco/annotations/instances_val2014.json
```
### Preprocess

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
python main.py generate_embeddings --encoder vit_b --checkpoint checkpoints/sam_vit_b_01ec64.pth --use_sam_checkpoint --directory data/coco/train_val_2017 --batch_size 16 --num_workers=8 --outfolder data/coco/vit_sam_embeddings/last_hidden_state --last_block_dir data/coco/vit_sam_embeddings/last_block_state --custom_preprocess
```

For ViT-MAE

```bash
python main.py generate_embeddings --encoder vit_b_mae --directory data/coco/train_val_2017 --batch_size 32 --num_workers 2 --outfolder data/coco/embeddings_vit_mae_1024/ --model_name facebook/vit-mae-base --image_resolution 1024 --mean_std default --huggingface

python main.py generate_embeddings --encoder vit_b_mae --directory data/coco/train_val_2017 --batch_size 64 --num_workers 2 --outfolder data/coco/embeddings_vit_mae_480 --model_name facebook/vit-mae-base --image_resolution 480 --mean_std default --huggingface

python main.py generate_embeddings --encoder vit_l_mae --directory data/coco/train_val_2017 --batch_size 64 --num_workers 2 --outfolder data/coco/embeddings_vit_mae_l_480 --model_name facebook/vit-mae-large --image_resolution 480 --mean_std default --huggingface
```

### Train and Test

You can train LabelAnything (ViT-MAE) model on COCO-20i by running the command:

```bash
python main.py experiment --parameters="parameters/coco20i/mae_noembs.yaml"
```

If you extracted the embeddings you can run the command:

```bash
python main.py experiment --parameters="parameters/coco20i/mae.yaml"
```

By default, four training processes will be launched sequentially, one for each fold of the 4-fold cross-validation. It is possible to launch only interesting training by deleting them from the `other_grids` section of the parameter file. Remember to also change the `val_fold_idx` in the `parameters.dataset` section to the fold you want to validate, which will be executed at the beginning. If you start a model training, you don't need to run the the validation step, as it is already included in the training process.

If you have a multi GPU machine, you can run the command:

```bash
accelerate launch --multi_gpu main.py experiment --parameters="parameters/COCO.yaml"
accelerate launch --multi_gpu main.py experiment --parameters="parameters/COCO_vit.yaml"  
```

Experiments are tracked using [Weights & Biases](https://wandb.ai/site). The resulting run files are stored in the `offline/wandb/run-<date>-<run_id>` directory. Model weights for the specific run are saved in the `files` subdirectory of the run folder.

## Project Organization

```
📦 Project Root
├── .gitignore               # Git exclusions
├── .python-version          # Python version lock
├── LICENSE                  # License file
├── README.md                # Project documentation
├── pyproject.toml           # Build system config
├── setup.py                 # Install script (setuptools)
├── main.py                  # Possibly main script or entry point
├── app.py                   # Alternative app entry point
├── test.py                  # Test runner or example test
├── uv.lock                  # Dependency lock file (for `uv`)

├── label_anything/          # 🔧 Core project code
│   ├── __main__.py          # CLI entry point
│   ├── cli.py               # Command-line interface
│   ├── data/                # Dataset loaders & preprocessing
│   ├── demo/                # Web demos (Streamlit, Gradio, NiceGUI)
│   ├── experiment/          # Training and experiment scripts
│   ├── logger/              # Logging tools (console, wandb, etc.)
│   ├── loss/                # Custom loss functions
│   ├── models/              # Model architectures and utilities
│   ├── utils/               # General helper functions
│   ├── visualization/       # Plotting and visual tools
│   ├── metrics.py           # Evaluation metrics
│   ├── preprocess.py        # Preprocessing logic
│   └── preprocess_clip.py   # CLIP-specific preprocessing

├── parameters/              # 📋 Training configuration (YAML)
│   ├── coco/                # COCO dataset configs
│   ├── pascal/              # Pascal VOC configs
│   ├── other/, ablations/   # Miscellaneous & ablation configs
│   └── old/                 # Legacy configs (for reference)
├── parameters_test/         # 🧪 Test-time configurations
├── parameters_validation/   # ✅ Validation experiments

├── notebooks/               # 📓 Jupyter notebooks
│   ├── demo.ipynb           # Demo notebook
│   ├── check_dataset.ipynb  # Dataset inspection
│   └── ...                  # Other dataset/model analysis notebooks

├── slurm/                   # ⚙️ HPC job scripts (SLURM)
│   ├── launch_run           # SLURM launcher scripts
│   ├── generate_embeddings  # Embedding extraction jobs
│   └── slurm.py             # Python SLURM utilities

├── assets/                  # 📁 Static assets (e.g., images)
│   └── la.png
├── data/                    # 📁 Data setup and placeholders
│   ├── .gitkeep             # Keeps the folder in git
│   └── script/              # Dataset setup scripts
└── checkpoints/             # 💾 Saved model checkpoints
```

