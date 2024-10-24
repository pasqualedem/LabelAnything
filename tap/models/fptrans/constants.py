import os
from pathlib import Path

project_dir = Path(__file__).parent
data_dir = project_dir / "data"
lists_dir = project_dir / "lists"

if 'RUN_POSITION' in os.environ and os.environ['RUN_POSITION'] == 'paddlecloud':
    on_cloud = True
    # Change these lines if you run the code on cloud
    pretrained_dir = project_dir / "afs/proj/FSS-PyTorch-ori/pretrained"
    output_dir = project_dir / "afs/proj/FSS-PyTorch-ori/output"
else:
    on_cloud = False
    # Change these lines if you run the code on localhost
    pretrained_dir = data_dir / "pretrained"
    output_dir = project_dir / "output"


pretrained_weights = {
    # "vgg16":        pretrained_dir / "vgg16-397923af.pth",
    # "resnet50":     pretrained_dir / "resnet50-19c8e357.pth",
    # "resnet101":    pretrained_dir / "resnet101-5d3b4d8f.pth",
    # "cyctr":        pretrained_dir / "cyctr_pascal_res50_split",
    # "pfenet":       pretrained_dir / "split",
    # "resnet50v2":   pretrained_dir / "resnet50_v2.pth",

    # "ViT-B/8":          pretrained_dir / "vit/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    # "ViT-B/16":         pretrained_dir / "vit/B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-B/16-384":     Path("checkpoints") / "vit/B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    # "ViT-B/16-i21k":    pretrained_dir / "vit/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    # "ViT-B/16-i21k-384":pretrained_dir / "vit/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    # "ViT-S/16":         pretrained_dir / "vit/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    # "ViT-S/16-i21k":    pretrained_dir / "vit/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    # "ViT-L/16":         pretrained_dir / "vit/L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    # "ViT-L/16-384":     pretrained_dir / "vit/L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",

    "DeiT-T/16":        pretrained_dir / "deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
    "DeiT-S/16":        pretrained_dir / "deit/deit_small_distilled_patch16_224-649709d9.pth",
    # "DeiT-B/16":        pretrained_dir / "deit/deit_base_distilled_patch16_224-df68dfff.pth",
    "DeiT-B/16-384":    pretrained_dir / "deit/deit_base_distilled_patch16_384-d0272ac0.pth",

    # "MiT-B3":           pretrained_dir / "mit/mit_b3.pth",
    # "MiT-B5":           pretrained_dir / "mit/mit_b5.pth",

    # "Swin-B/16-384":    pretrained_dir / "swin/swin_base_patch4_window12_384.pth"

}

model_urls = {
    # 'resnet50':         'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'resnet101':        'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

    # "ViT-B/8":          "https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    # "ViT-B/16":         "https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "ViT-B/16-384":     "https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    # "ViT-B/16-i21k":    "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    # "ViT-B/16-i21k-384":"https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    # "ViT-S/16":         "https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    # "ViT-S/16-i21k":    "https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    # "ViT-L/16":         "https://storage.googleapis.com/vit_models/augreg/L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    # "ViT-L/16-384":     "https://storage.googleapis.com/vit_models/augreg/L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",

    "DeiT-T/16":        "https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
    "DeiT-S/16":        "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
    # "DeiT-B/16":        "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
    "DeiT-B/16-384":    "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",

}
