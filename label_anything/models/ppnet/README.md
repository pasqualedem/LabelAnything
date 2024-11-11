# PPNet - Part-Aware Prototype Network

## Train PPNet

PPNet is trained using the original repository. Model weights are saved as `best.pth` files in fold-specific directories in the original repo.

## Test PPNet

Move the directory containing the `best.pth` file in this repository. Specify the path as the `ckpt_dir` in the yaml file. Then put ResNet weights in the dir named `resnet` under the parent of the model directory. Specify the `fold` in the yaml file. Use only one test dataset, since more test dataset would require train a specific model for any combinations of N and K.