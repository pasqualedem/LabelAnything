import tqdm
import torch.nn as nn
from torch.optim import Adam
import torch
import comet_ml
from comet_ml.integration.pytorch import log_model
import os
from logger import logger

comet_information = {
    'apykey': os.getenv('COMET_API_KEY'),
    'project_name': 'cv_pasquale_il_mastodontico'
}

comet_ml.init(comet_information)

experiment = comet_ml.Experiment()
experiment_name = 'random-name'
hyper_params = {"batch_size": 100, "num_epochs": 2, "learning_rate": 0.01}

experiment.log_parameters(hyper_params)
experiment.add_tags([experiment_name])


def get_train_data(batch_dict: dict):
    """Convert the dictionary into the elements for training

    Args:
        batch_dict (dict): data dictionary with the training data

    Returns:
        target: The target tensor.
        example: The example tensor.
        prompt_mask: The prompt_mask tensor.
        prompt_point: The prompt_point tensor.
        prompt_bbox: The prompt_bbox tensor.
        gt: The gt tensor.
    """
    # Per ogni chiave in batch_dict, sposta il tensor corrispondente sul dispositivo e lo memorizza nel dizionario output
    output = {}
    for key in batch_dict:
        output[key] = batch_dict[key].to(device)

    # Estrae i tensori dal dizionario output
    target, example, p_mask, p_point, p_bbox, gt = output.values()
    return target, example, p_mask, p_point, p_bbox, gt


def train(model, optimizer, criterion, dataloader, epoch, experiment):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        target, example, p_mask, p_point, p_bbox, gt = get_train_data(
            batch_dict)

        outputs = model(target, example, p_mask, p_point, p_bbox)
        loss = criterion(outputs, gt)

        # TODO change pred argmax??
        pred = outputs.argmax(dim=1, keepdim=True)

        loss.backward()
        optimizer.step()

        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        batch_total = target.size(0)

        total_loss += loss.item()
        correct += batch_correct

        # Log batch_accuracy to Comet; step is each batch
        experiment.log_metric("batch_accuracy", batch_correct / batch_total)

    total_loss /= len(dataloader.dataset)
    correct /= len(dataloader.dataset)

    experiment.log_metrics(
        {"accuracy": correct, "loss": total_loss}, epoch=epoch)


# TODO: inserisci qui il modello che deve essere lanciato
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

#TODO: cosa sarà il modello? Un argomento? Probably
model = Model()

# TODO: dataset e dataloader che cazzo mettiamo?
# dataset
# dataloader

if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=hyper_params['learning_rate'])

# Train the Model
with experiment.train():
    logger.info(f"Running Model Training {experiment_name}")
    for epoch in range(hyper_params["num_epochs"]):
        train(model, optimizer, criterion, dataloader, epoch, experiment)

log_model(experiment, model, "mnist-classifier")
experiment.end()
