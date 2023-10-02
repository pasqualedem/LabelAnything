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


def train(model, optimizer, criterion, dataloader, epoch, experiment):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        target = batch_dict["target"].to(device)
        example = batch_dict["example"].to(device)
        p_mask = batch_dict["p_mask"].to(device)
        p_point = batch_dict["p_point"].to(device)
        p_bbox = batch_dict["p_bbox"].to(device)
        gt = batch_dict["gt"].to(device)

        # TODO cosa passo al modello?
        outputs = model(images)

        loss = criterion(outputs, labels)
        pred = outputs.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability

        loss.backward()
        optimizer.step()

        # Compute train accuracy
        batch_correct = pred.eq(labels.view_as(pred)).sum().item()
        batch_total = labels.size(0)

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
model = Model()

if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=hyper_params["learning_rate"])

# Train the Model
with experiment.train():
    print("Running Model Training")
    for epoch in range(hyper_params["num_epochs"]):
        train(model, optimizer, criterion, dataloader, epoch, experiment)

log_model(experiment, model, "mnist-classifier")
experiment.end()
