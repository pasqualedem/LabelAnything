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


@main.command("preprocess_huggingface")
@click.option(
    "--model_name",
    default="facebook/vim-mae-base",
    help="Select model to use",
)
@click.option(
    "--compile",
    is_flag=True,
    help="Select if the model should be compiled",
)
@click.option(
    "--directory",
    default="/leonardo_scratch/large/userexternal/nfanelli/train_val_2017",
    help="Select the file to use as checkpoint",
)
@click.option(
    "--batch_size",
    default=16,
    help="Batch size for the dataloader",
)
@click.option(
    "--num_workers",
    default=8,
    help="Number of workers for the dataloader",
)
@click.option(
    "--outfolder",
    default="/leonardo_scratch/large/userexternal/rscaring/vit_embeddings",
    help="Folder to save the embeddings",
)
def preprocess_huggingface(
        model_name,
        directory,
        batch_size,
        num_workers,
        outfolder,
        compile,
):
    from label_anything.preprocess import preprocess_images_to_embeddings_huggingface
    preprocess_images_to_embeddings_huggingface(
        model_name=model_name,
        directory=directory,
        batch_size=batch_size,
        num_workers=num_workers,
        outfolder=outfolder,
        compile=compile,
    )


@main.command("preprocess")
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
def preprocess(
    encoder,
    checkpoint,
    use_sam_checkpoint,
    compile,
    directory,
    batch_size,
    num_workers,
    outfolder,
):
    from label_anything.preprocess import preprocess_images_to_embeddings

    preprocess_images_to_embeddings(
        encoder_name=encoder,
        checkpoint=checkpoint,
        use_sam_checkpoint=use_sam_checkpoint,
        directory=directory,
        batch_size=batch_size,
        num_workers=num_workers,
        outfolder=outfolder,
        compile=compile,
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


@main.command("pretrain_pe")
@click.option(
    "--parameters", default="pretraining_parameters.yaml", help="Path to yaml file"
)
def pretrain_pe(parameters):
    exe_pretrain_pe(parameters)


@main.command("rename_coco20i_json")
@click.option(
    "--instances_path",
    default="data/raw/instances_train2014.json",
    help="Path to the instances file",
)
def rename_coco20i_json_cli(instances_path):
    from label_anything.preprocess import rename_coco20i_json
    rename_coco20i_json(instances_path)
