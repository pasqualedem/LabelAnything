import tqdm
import torch

from torch.optim import Adam
from label_anything.logger.utils import (
    extract_boxes_from_tensor,
    extract_vertices_from_tensor,
    image_with_points,
    structure_annotations,
)
from save import save_model
from utils.utils import log_every_n
from logger.text_logger import get_logger
from accelerate import Accelerator
from torchmetrics import JaccardIndex


logger = get_logger(__name__)


def train_epoch(
    args,
    model,
    optimizer,
    criterion,
    dataloader,
    epoch,
    comet_logger,
    accelerator,
    train_metrics,
):
    model.train()
    total_loss = 0
    correct = 0
    total_jaccard = 0

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    jaccard = JaccardIndex(task="multiclass", num_classes=outputs.shape[1])

    for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        image_dict, gt = batch_dict

        outputs = model(image_dict)
        loss = criterion(outputs, gt)

        pred = outputs.argmax(dim=1, keepdim=True)
        jaccard_val = jaccard(pred, gt)

        accelerator.backward()
        optimizer.step()

        batch_correct = pred.eq(image_dict["example"].view_as(pred)).sum().item()
        batch_total = image_dict["example"].size(0)

        total_loss += loss.item()
        correct += batch_correct
        comet_logger.log_metric("batch_accuracy", batch_correct / batch_total)
        comet_logger.log_metric("batch_jaccard", jaccard_val.item())

        if log_every_n(batch_idx, args.logger["n_iter"]):
            query_image = image_dict["query_image"][0]
            points = image_dict["prompt_points"][0, 0]
            boxes = image_dict["prompt_boxes"][0, 0]
            mask = image_dict["prompt_mask"][0, 0]
            annotations_boxes = structure_annotations(extract_boxes_from_tensor(boxes))
            annotations_mask = structure_annotations(extract_vertices_from_tensor(mask))
            comet_logger.log_image(
                query_image,
                annotations_mask,
            )
            comet_logger.log_image(
                query_image,
                annotations_boxes,
            )
            comet_logger.log_image(image_with_points(query_image, points))

    total_loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)

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


def train(args, model, dataloader, comet_logger, experiment, hyper_params):
    logger.info("Start run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    accelerator = Accelerator()

    # Loss and Optimizer da overraidere
    criterion = hyper_params["loss"]
    train_metrics = hyper_params["train_metrics"]
    optimizer = Adam(model.parameters(), lr=hyper_params["learning_rate"])

    logger.info("CIOLA")
    # Train the Model
    with experiment.train():
        logger.info(f"Running Model Training {args.name}")
        for epoch in range(hyper_params["num_epochs"]):
            logger.info("Epoch: {}/{}".format(epoch, hyper_params["num_epochs"]))
            train_epoch(
                args,
                model,
                optimizer,
                criterion,
                dataloader,
                epoch,
                comet_logger,
                accelerator,
                train_metrics,
            )

    save_model(experiment, model, model._get_name)
    logger.info(f"Finished Training {args.name}")

    # with experiment.test():
    #     logger.info(f"Running Model Testing {args.name}")
    #     for epoch in range(hyper_params["num_epochs"]):
    #         test(args, model, criterion, dataloader, comet_logger)

    # logger.info(f"Finished Testing {args.name}")
    experiment.end()
