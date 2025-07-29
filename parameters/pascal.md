Setting up [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) Dataset with augmented data.

### 1. Instruction to download
``` bash
bash data/script/setup_voc12.sh data/pascal
``` 
```bash
data/
└── pascal/
    ├── Annotations
    ├── ImageSets/
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationObject
    └── SegmentationClass
``` 
### 2. Add SBD Augmentated training data
- Convert by yourself ([here](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal)).
- Or download pre-converted files ([here](https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation)), **(Prefer this method)**.

After the download move it into the pascal folder.

```bash
unzip SegmentationClassAug.zip -d data/pascal
```

```bash
data/
└── pascal/
    ├── Annotations
    ├── ImageSets/
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationObject
    ├── SegmentationClass
    └── SegmentationClassAug #ADDED
``` 

### 3. Download official sets as ImageSets/SegmentationAug list
From: https://github.com/kazuto1011/deeplab-pytorch/files/2945588/list.zip

```bash
# Unzip the file
unzip list.zip -d data/pascal/ImageSets/
# Move file into Segmentation folder
mv data/pascal/ImageSets/list/* data/pascal/ImageSets/Segmentation/
rm -rf data/pascal/ImageSets/list
```

This is how the dataset should look like
```bash
/data
└── pascal
    ├── Annotations
    ├── ImageSets
    │   └── Segmentation 
    │       ├── test.txt
    │       ├── trainaug.txt # ADDED!!
    │       ├── train.txt
    │       ├── trainvalaug.txt # ADDED!!
    │       ├── trainval.txt
    │       └── val.txt
    ├── JPEGImages
    ├── SegmentationObject
    ├── SegmentationClass
    └── SegmentationClassAug # ADDED!!
        └── 2007_000032.png
```
### 4. Rename
Now run the rename.sh script.
``` bash
bash data/script/rename.sh data/pascal/ImageSets/Segmentation/train.txt
bash data/script/rename.sh data/pascal/ImageSets/Segmentation/trainval.txt
bash data/script/rename.sh data/pascal/ImageSets/Segmentation/val.txt
``` 

## Generate Embeddings

```bash
mkdir -p data/pascal/vit_sam_embeddings/last_hidden_state
mkdir data/pascal/vit_sam_embeddings/last_block_state
python main.py generate_embeddings --encoder vit_b --checkpoint checkpoints/sam_vit_b_01ec64.pth --use_sam_checkpoint --directory data/pascal/JPEGImages --batch_size 16 --num_workers=8 --outfolder data/pascal/pascal_embeddings_vit_b_sam/last_hidden_state --last_block_dir data/pascal/pascal_embeddings_vit_b_sam/last_block_state --custom_preprocess

python main.py generate_embeddings --encoder vit_b_mae --directory data/pascal/JPEGImages --batch_size 64 --num_workers 8 --outfolder data/pascal/embeddings_vit_mae_480 --model_name facebook/vit-mae-base --image_resolution 480 --mean_std default --huggingface
```