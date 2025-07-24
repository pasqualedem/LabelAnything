from label_anything.experiment.experiment import (
    experiment as run_experiment,
    run as run_single,
    test as test_fn,
    validate as validate_fn,
)

import click
from label_anything.experiment.pretraining import main as exe_pretrain_pe


@click.group()
def main():
    pass


@main.command("experiment")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
@click.option(
    "--parallel", default=False, help="Run the experiments in parallel", is_flag=True
)
@click.option(
    "--only-create",
    default=False,
    help="Creates params files with running them",
    is_flag=True,
)
def experiment(parameters, parallel, only_create):
    run_experiment(param_path=parameters, parallel=parallel, only_create=only_create)


@main.command("run")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
def run(parameters):
    run_single(param_path=parameters)


@main.command("test")
@click.option("--parameters", default="test.yaml")
def test(parameters):
    test_fn(param_path=parameters)


@main.command("validate")
@click.option("--parameters", default="test.yaml")
def validate(parameters):
    validate_fn(param_path=parameters)


@main.command("generate_embeddings")
@click.option(
    "--encoder",
    default="vit_h",
    help="Select the encoder to use",
)
@click.option(
    "--checkpoint",
    default="vit_h.pth",
    help="Select the file to use as checkpoint",
)
@click.option(
    "--use_sam_checkpoint",
    is_flag=True,
    help="Select if the checkpoint is a SAM checkpoint",
)
@click.option(
    "--compile",
    is_flag=True,
    help="Select if the model should be compiled",
)
@click.option(
    "--directory",
    default="data/raw/train2017",
    help="Select the file to use as checkpoint",
)
@click.option(
    "--batch_size",
    default=1,
    help="Batch size for the dataloader",
)
@click.option(
    "--num_workers",
    default=0,
    help="Number of workers for the dataloader",
)
@click.option(
    "--outfolder",
    default="data/processed/embeddings",
    help="Folder to save the embeddings",
)
@click.option(
    "--device",
    default="cuda",
    help="Device to use for the model",
)
@click.option(
    "--last_block_dir",
    default=None,
    help="Folder to save last transformer block",
)
@click.option(
    "--custom_preprocess",
    is_flag=True,
    help="Whether to use custom resize and normalize",
)
@click.option(
    "--huggingface",
    is_flag=True,
    help="Whether to use huggingface models",
)
@click.option(
    "--model_name",
    default="facebook/vit-mae-base",
    help="Select model to use (Only for huggingface models)",
)
@click.option(
    "--image_resolution",
    default=480,
    help='Image resolution for ViT (Only for huggingface models)',
)
@click.option(
    "--mean_std",
    default="default",
    help="Mean and std for normalization (can be default or standard) (Only for huggingface models)",
)
def generate_embeddings(
    encoder,
    checkpoint,
    use_sam_checkpoint,
    compile,
    directory,
    batch_size,
    num_workers,
    outfolder,
    device,
    last_block_dir,
    custom_preprocess,
    huggingface,
    model_name,
    image_resolution,
    mean_std,
):

    if huggingface:
        from label_anything.preprocess import preprocess_images_to_embeddings_huggingface
        preprocess_images_to_embeddings_huggingface(
            model_name=model_name,
            directory=directory,
            batch_size=batch_size,
            num_workers=num_workers,
            outfolder=outfolder,
            device=device,
            compile=compile,
            image_resolution=image_resolution,
            custom_preprocess=custom_preprocess,
            mean_std=mean_std,
        )
    else:
        from label_anything.preprocess import preprocess_images_to_embeddings
        preprocess_images_to_embeddings(
            encoder_name=encoder,
            checkpoint=checkpoint,
            use_sam_checkpoint=use_sam_checkpoint,
            directory=directory,
            batch_size=batch_size,
            num_workers=num_workers,
            outfolder=outfolder,
            last_block_dir=last_block_dir,
            compile=compile,
            custom_preprocess=custom_preprocess,
        )


@main.command("generate_feature_pyramids")
@click.option(
    "--encoder_name",
    default="resnet50",
    help="Select the encoder to use",
)
@click.option(
    "--directory",
    default="data/raw/train2017",
    help="Select the file to use as checkpoint",
)
@click.option(
    "--batch_size",
    default=1,
    help="Batch size for the dataloader",
)
@click.option(
    "--num_workers",
    default=0,
    help="Number of workers for the dataloader",
)
@click.option(
    "--outfolder",
    default="data/processed/embeddings",
    help="Folder to save the embeddings",
)
@click.option(
    "--device",
    default="cuda",
    help="Device to use for the model",
)
@click.option(
    "--compile",
    is_flag=True,
    help="Select if the model should be compiled",
)
@click.option(
    "--image_resolution",
    default=384,
    help="Image resolution for the model",
)
@click.option(
    "--custom_preprocess",
    is_flag=True,
    help="Whether to use custom resize and normalize",
)
@click.option(
    "--out_features",
    default="stage2,stage3,stage4",
    help="Output features to use",
)
@click.option(
    "--mean_std",
    default="default",
    help="Mean and std for normalization (can be default or standard)",
)
def generate_feature_pyramids(
    encoder_name,
    directory,
    batch_size,
    num_workers,
    outfolder,
    device,
    compile,
    image_resolution,
    custom_preprocess,
    out_features,
    mean_std,
):
    out_features = out_features.split(",")

    from label_anything.preprocess import preprocess_images_to_feature_pyramids
    preprocess_images_to_feature_pyramids(
        encoder_name=encoder_name,
        directory=directory,
        batch_size=batch_size,
        num_workers=num_workers,
        outfolder=outfolder,
        device=device,
        compile=compile,
        image_resolution=image_resolution,
        custom_preprocess=custom_preprocess,
        out_features=out_features,
        mean_std=mean_std,
    )
    

@main.command("generate_gt")
@click.option(
    "--dataset_name",
    default="coco",
    help="Select the dataset to use",
)
@click.option(
    "--anns_path",
    default="data/raw/instances_train2017.json",
    help="Select the file to use as checkpoint",
)
@click.option(
    "--outfolder",
    default="embeddings",
    help="Folder to save the embeddings",
)
def generate_gt(dataset_name, anns_path, outfolder):
    from label_anything.preprocess import generate_ground_truths

    generate_ground_truths(dataset_name, anns_path, outfolder)


@main.command("benchmark")
def benchmark():
    import torch
    import torch.nn as nn
    import time

    # Define a simple neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(1000, 500)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Create an instance of the network
    net = SimpleNet().cuda()

    # Generate random input data
    input_data = torch.randn(100, 1000).cuda()

    # Warm-up GPU
    for _ in range(10):
        _ = net(input_data)

    # Benchmark the forward pass on GPU
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        _ = net(input_data)
    end_time = time.time()

    # Calculate average time per iteration
    average_time = (end_time - start_time) / num_iterations

    print(f"Average time per iteration: {average_time:.5f} seconds")


@main.command("preprocess_clip")
@click.option("--parameters", default="extract_params.yaml", help="Path to yaml file")
def preprocess_clip(parameters):
    from label_anything.preprocess_clip import main as exe_clip_preprocess

    exe_clip_preprocess(params_path=parameters)


@main.command("preprocess_voc")
@click.option(
    "--input_folder",
    default=None,
    help="Path to the VOC dataset",
)
def preprocess_voc(input_folder):
    from label_anything.data.voc12 import preprocess_voc as preprocess_voc_fn
    preprocess_voc_fn(input_folder)


@main.command("pretrain_pe")
@click.option(
    "--parameters", default="pretraining_parameters.yaml", help="Path to yaml file"
)
def pretrain_pe(parameters):
    exe_pretrain_pe(parameters)


@main.command("rename_coco20i_json")
@click.option(
    "--instances_path",
    default="data/annotations/instances_train2014.json",
    help="Path to the instances file",
)
def rename_coco20i_json_cli(instances_path):
    from label_anything.preprocess import rename_coco20i_json

    rename_coco20i_json(instances_path)

@main.command("app")
def app():
    from label_anything.demo.nicegui import main as nicegui_main
    nicegui_main()