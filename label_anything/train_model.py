import tqdm

# TODO: remove and override by parameters
from torch.optim import Adam
import torch
from save import save_model
from utils.utils import log_every_n
from label_anything.logger.logger import logger


def train(args, model, optimizer, criterion, dataloader, epoch, comet_logger):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        image_dict, gt = batch_dict

        # TODO: flag bbox parla con Pasq
        outputs = model(image_dict)
        loss = criterion(outputs, gt)

        pred = outputs.argmax(dim=1, keepdim=True)

        loss.backward()
        optimizer.step()

        batch_correct = pred.eq(image_dict["example"].view_as(pred)).sum().item()
        batch_total = image_dict["example"].size(0)

        total_loss += loss.item()
        correct += batch_correct
        comet_logger.log_metric("batch_accuracy", batch_correct / batch_total)
        if log_every_n(batch_idx, args.logger["n_iter"]):
            big_logger()

    total_loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)

    comet_logger.log_metrics({"accuracy": correct, "loss": total_loss}, epoch=epoch)


def test(args, model, criterion, dataloader, epoch, comet_logger):
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


def run(args, model, dataloader, comet_logger, experiment, hyper_params):
    logger.info("Running")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Loss and Optimizer da overraidere
    criterion = hyper_params["loss"]
    optimizer = Adam(model.parameters(), lr=hyper_params["learning_rate"])

    # Train the Model
    with experiment.train():
        logger.info(f"Running Model Training {args.name}")
        for epoch in range(hyper_params["num_epochs"]):
            logger.info("Epoch: {}/{}".format(epoch, hyper_params["num_epochs"]))
            train(args, model, optimizer, criterion, dataloader, epoch, comet_logger)

    save_model(experiment, model, model._get_name)
    logger.info(f"Finished Training {args.name}")

    with experiment.test():
        logger.info(f"Running Model Testing {args.name}")
        for epoch in range(hyper_params["num_epochs"]):
            test(args, model, criterion, dataloader, comet_logger)

    logger.info(f"Finished Testing {args.name}")
    experiment.end()


# DEPRECATED
def get_single_data(batch_dict: dict, device):
    """Convert the dictionary into the elements for training

    Args:
        batch_dict (dict): data dictionary with the training data

    Returns:
        target: The target tensor.
        example: The example tensor.
        prompt_mask: The prompt_mask tensor.
        prompt_point: The prompt_point tensor.
        prompt_bbox: The prompt_bbox tensor.
        flag_bbox: The flag_bbox tensor.
        gt: The gt tensor.
    """
    # Per ogni chiave in batch_dict, sposta il tensor corrispondente sul dispositivo e lo memorizza nel dizionario output
    output = {}
    for key in batch_dict:
        output[key] = batch_dict[key].to(device)

    # Estrae i tensori dal dizionario output
    target, example, p_mask, p_point, p_bbox, flag_bbox, gt = output.values()
    return target, example, p_mask, p_point, p_bbox, flag_bbox, gt
