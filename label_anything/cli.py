import comet_ml
from label_anything.preprocess import preprocess_images_to_embeddings
from label_anything.experiment.experiment import experiment as run_experiment

import click


@click.group()
def main():
    pass


@main.command("experiment")
@click.option("--parameters", default="parameters.yaml", help="Path to the parameters file")
def experiment(params):
    run_experiment(param_path=params)


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