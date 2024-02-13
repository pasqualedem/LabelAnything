import random

import accelerate
from label_anything.models.contrastive_pe import ContrastivePromptEncoder, PromptImageEncoder
from torch.utils.data import DataLoader
from label_anything.loss.symmetric import SymmetricLoss
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from accelerate import Accelerator
from label_anything.data.prompt_encoder_dataset import PromptEncoderDataset
import torch
from label_anything.utils.early_stopping import ParallelEarlyStopping
from tqdm import tqdm
from label_anything.models import model_registry
from label_anything.preprocess_clip import load_ruamel
from label_anything.data.prompt_encoder_dataset import collate_fn


def change_num_examples(train_loader: DataLoader,
                        val_loader: DataLoader,
                        n_examples):
    train_loader.dataset.set_num_examples(n_examples)
    val_loader.dataset.set_num_examples(n_examples)


def train(
        model: ContrastivePromptEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: SymmetricLoss,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        accelerator: Accelerator,
        early_stop: ParallelEarlyStopping,
        num_epochs: int,
        min_num_examples: int,
        max_num_examples: int,
):
    loaders = {'train': train_loader, 'val': val_loader}
    optimizer.zero_grad()
    for epoch in range(1, num_epochs + 1):
        for phase, loader in loaders.items():
            accelerator.print(f'Phase: {phase}')
            cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)
            with torch.set_grad_enabled(phase == 'train'):
                for data_dict in tqdm(loader):
                    prompt_proj, clip_proj = model(data_dict)
                    label = torch.eye(prompt_proj.size(0)).to(accelerator.device)
                    loss = criterion(prompt_proj, clip_proj, label)
                    cumulated_loss = cumulated_loss + loss
                    if phase == 'train':
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
            if phase == 'val':
                cumulated_loss = accelerator.gather(cumulated_loss)
                if accelerator.is_main_process:
                    cumulated_loss = cumulated_loss.cpu().mean().item()
                    accelerator.print(f'Epoch {epoch:04d}: Val loss: {cumulated_loss:.4f}')
                    early_stop(cumulated_loss, accelerator)
                    scheduler.step(cumulated_loss)
                    if early_stop.early_stop:
                        accelerator.set_trigger()
                        accelerator.print(f'early stopping at epoch {epoch:03d}')
        accelerator.wait_for_everyone()
        if accelerator.check_trigger():
            break
        n_examples = random.randint(min_num_examples, max_num_examples)
        change_num_examples(train_loader, val_loader, n_examples)


def init_model(model_params: dict) -> dict:
    model_name = model_params.get('prompt_encoder').pop('name')
    lam = model_registry[model_name](**model_params['prompt_encoder'])
    prompt_encoder = lam.prompt_encoder
    model_params['prompt_encoder'] = prompt_encoder
    return model_params


def main(params_path):
    print('Load parameters...')
    params = load_ruamel(params_path)
    print('Done!')

    print('Initializing model...')
    params['model'] = init_model(params['model'])
    model = ContrastivePromptEncoder(**params['model'])
    print('Done!')

    print('Initializing training data...')
    train_data = PromptEncoderDataset(**params['dataset']['train'])
    train_loader = DataLoader(dataset=train_data, collate_fn=collate_fn, **params['dataloader'])
    print('Done"')

    print('Initializing validation data...')
    val_data = PromptEncoderDataset(**params['dataset']['val'])
    val_loader = DataLoader(dataset=val_data, collate_fn=collate_fn, **params['dataloader'])
    print('Done!')

    print('Initializing criterion...')
    criterion = SymmetricLoss(**params.get('criterion', {}))
    print('Done!')

    print('Initializing optimizer...')
    optimizer = AdamW(params=model.parameters(), **params.get('optimizer', {}))
    print('Done!')

    print('Initializing scheduler...')
    scheduler = ReduceLROnPlateau(optimizer=optimizer, **params.get('scheduler', {}))
    print('DOne!')

    print('Initializing early stop mechanism...')
    early_stop = ParallelEarlyStopping(**params['early_stopping'])
    print('Done!')

    print('Initializing accelerator...')
    kwargs = [accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs, **params.get('accelerator', {}))
    print('Done!')

    if accelerator.is_main_process:
        print('Starting training!')
    (
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        model,
    ) = accelerator.prepare(
        train_loader, val_loader, criterion, optimizer, scheduler, model,
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        early_stop=early_stop,
        **params['train_loop']
    )

