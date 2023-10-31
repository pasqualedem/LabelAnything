import comet_ml
from label_anything.preprocess import preprocess_images_to_embeddings
from label_anything.experiment.experiment import experiment as run_experiment

import click


@click.group()
def main():
    pass


@main.command("experiment")
def experiment():
    run_experiment()


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
