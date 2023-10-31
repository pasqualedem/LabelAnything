from tqdm import tqdm
import torch

from torch.optim import AdamW
from label_anything.logger.text_logger import get_logger
from label_anything.logger.utils import (
    extract_boxes_from_tensor,
    extract_vertices_from_tensor,
    image_with_points,
    structure_annotations,
)
from label_anything.experiment.substitution import Substitutor
from label_anything.utils.utils import log_every_n
from .save import save_model
from accelerate import Accelerator
from torchmetrics.functional.classification import multiclass_jaccard_index

logger = get_logger(__name__)


def train_epoch(
    model,
    optimizer,
    criterion,
    dataloader,
    epoch,
    comet_logger,
    accelerator,
    train_params,
):
    model.train()
    total_loss = 0
    correct = 0
    total_jaccard = 0

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
        substitutor = Substitutor(
            batch_dict,
            threshold=train_params["substitution_threshold"],
            num_points=train_params.get("num_points", 1),
        )
        for i, (input_dict, gt) in enumerate(substitutor):
            optimizer.zero_grad()

            outputs = model(input_dict)
            loss = criterion(outputs, gt)
            substitutor.generate_new_points(outputs, gt)

            pred = outputs.argmax(dim=1)
            jaccard = multiclass_jaccard_index(pred, gt, num_classes=outputs.shape[1])

            accelerator.backward(loss)
            optimizer.step()

            batch_total = gt.size(0)

            total_loss += loss.item()
            total_jaccard += jaccard.item()
            comet_logger.log_metric("batch_jaccard", jaccard.item())

            if log_every_n(batch_idx, train_params["logger"]):
                comet_logger.log_batch(
                    batch_idx=batch_idx,
                    step=i,
                    input_dict=input_dict,
                    categories=dataloader.dataset.categories,
                )

    total_loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)
    total_jaccard /= len(dataloader.dataset)

    comet_logger.log_metrics(
        {"accuracy": correct, "loss": total_loss, "jaccard": total_jaccard},
        epoch=epoch,
    )


def test(model, criterion, dataloader, epoch, comet_logger):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
            image_dict, gt = batch_dict

            output = model(image_dict)
            total_loss += criterion(output, gt).item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True,
            )  # get the index of the max log-probability
            correct += pred.eq(image_dict["example"].view_as(pred)).sum().item()

        total_loss /= len(dataloader.dataset)
        correct /= len(dataloader.dataset)

        comet_logger.log_metrics({"accuracy": correct, "loss": total_loss}, epoch=epoch)


def train(args, model, dataloader, comet_logger, experiment, train_params):
    logger.info("Start training loop...")
    accelerator = Accelerator()

    # Loss and Optimizer da overraidere
    criterion = lambda x, y: torch.nn.functional.cross_entropy(x, y.long())
    optimizer = AdamW(model.parameters(), lr=train_params["initial_lr"])

    # Train the Model
    with experiment.train():
        logger.info(f"Running Model Training {args.get('experiment').get('name')}")
        for epoch in range(train_params["max_epochs"]):
            logger.info("Epoch: {}/{}".format(epoch, train_params["max_epochs"]))
            train_epoch(
                model,
                optimizer,
                criterion,
                dataloader,
                epoch,
                comet_logger,
                accelerator,
                train_params,
            )

    # save_model(experiment, model, model._get_name)
    logger.info(f"Finished Training {args.get('experiment').get('name')}")

    # with experiment.test():
    #     logger.info(f"Running Model Testing {args.name}")
    #     for epoch in range(hyper_params["num_epochs"]):
    #         test(args, model, criterion, dataloader, comet_logger)

    # logger.info(f"Finished Testing {args.name}")
    experiment.end()