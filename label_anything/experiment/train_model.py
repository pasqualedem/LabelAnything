import gc
from tqdm import tqdm
import torch

from torch.optim import AdamW
from label_anything.logger.text_logger import get_logger
from label_anything.logger.image_logger import Logger
from label_anything.experiment.substitution import Substitutor
from label_anything.utils.utils import find_divisor_pairs, RunningAverage
from label_anything.data.utils import random_batch
from label_anything.loss import LabelAnythingLoss
from .save import save_model
from accelerate import Accelerator
from label_anything.utils.metrics import jaccard, fbiou
from transformers import get_scheduler

logger = get_logger(__name__)


def get_batch_size(batch_tuple):
    if batch_tuple[0].get("images") is not None:
        return batch_tuple[0]["images"].shape[0]
    if batch_tuple[0].get("embeddings") is not None:
        return batch_tuple[0]["embeddings"].shape[0]
    
    
def check_nan(model, input_dict, output, gt, loss, step, train_params):
    if not train_params.get("check_nan", False):
        return
    if step % train_params['check_nan'] != 0:
        return
    if torch.isnan(loss) or loss.detach() in [torch.inf, -torch.inf]:
        if train_params['check_nan'] == 1: # Makes sense only if we are checking every step
            state_dict = {
                "model": model.state_dict(),
                "input_dict": input_dict,
                "loss": loss,
                "step": step,
                "gt": gt,
                "output": output,
            }
            torch.save(state_dict, "nan.pt")
        raise ValueError("NaNs in loss")
    
    
def handle_oom(model, input_dict, batch_tuple, optimizer, gt, epoch, step):
    logger.warning(f"OOM at step {step}")
    logger.warning(torch.cuda.memory_summary())
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "input_dict": input_dict,
            "batch_tuple": batch_tuple,
            "gt": gt,
        },
        f"oom_epoch_{epoch}_step_{step}.pt",
    )
    optimizer.zero_grad()
    del input_dict
    del batch_tuple
    del gt
    gc.collect()
    torch.cuda.empty_cache()

def allocate_memory(model, accelerator, optimizer, criterion, dataloader):
    """
    Execute forward and backward with maximum input lenght to avoid out of memories
    """
    depth = 256
    height = 1024
    width = 1024
    num_classes = 10
    num_objects = 10

    max_num_images = dataloader.batch_sampler.get_max_num_images()
    batch_size_examples_pairs = find_divisor_pairs(max_num_images)
    logger.info(f"Max number of images: {max_num_images}")
    for batch_size, num_examples in batch_size_examples_pairs:
        optimizer.zero_grad()
        batch_dict, gt = random_batch(
            batch_size, num_examples, depth, height, width, num_classes, num_objects
        )
        batch_dict = accelerator.prepare(batch_dict)  # TODO: Make this work
        outputs = model(batch_dict)
        loss = criterion(outputs, gt)
        pred = outputs.argmax(dim=1)
        accelerator.backward(loss)
        optimizer.step()
        logger.info(f"Batch size {batch_size}; num examples: {num_examples} OK")
    logger.info("Allocating memory test PASSED")
    logger.info(torch.cuda.mem_get_info())


def train_epoch(
    model,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    epoch,
    comet_logger: Logger,
    accelerator: Accelerator,
    train_params,
):
    model.train()
    avg_loss = RunningAverage()
    avg_jaccard = RunningAverage()
    avg_fbiou = RunningAverage()
    first_step_loss = RunningAverage()
    first_step_jaccard = RunningAverage()
    first_step_fbiou = RunningAverage()

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    # allocate_memory(model, accelerator, optimizer, criterion, dataloader)

    bar = tqdm(enumerate(dataloader), total=len(dataloader), postfix={"loss": 0})
    tot_steps = 0
    tot_images = 0
    loss_normalizer = 1
    oom = False
    cur_lr = train_params["initial_lr"]
    dataloader.dataset.log_images = True # Logs the first batch

    for batch_idx, batch_tuple in bar:
        if batch_idx == 3:
            break
        batch_tuple, dataset_names = batch_tuple
        cur_batch_size = get_batch_size(batch_tuple)
        substitutor = Substitutor(
            batch_tuple,
            threshold=train_params.get("substitution_threshold", None),
            num_points=train_params.get("num_points", 1),
            substitute=train_params.get("substitute", True),
        )
        loss_normalizer = batch_tuple[1].shape[1] if train_params.get("accumulate_substitution", True) else 1
        for i, (input_dict, gt) in enumerate(substitutor):
            try:
                outputs = model(input_dict)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    handle_oom(model, input_dict, batch_tuple, optimizer, gt, epoch, batch_idx)
                    if oom:
                        raise e
                    oom = True
                    break
                else:
                    raise e
            loss = criterion(outputs, gt) / loss_normalizer
            accelerator.backward(loss)
            oom = False
            pred = outputs.argmax(dim=1)
            check_nan(model, input_dict, outputs, gt, loss, batch_idx, train_params)
            if not train_params.get("accumulate_substitution", False) or i == loss_normalizer - 1:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            if tot_steps % comet_logger.log_frequency == 0:
                pred = accelerator.gather(pred)
                gt = accelerator.gather(gt)
                outputs = accelerator.gather(outputs)
                jaccard_value = jaccard(pred, gt, num_classes=outputs.shape[1])
                fbiou_value = fbiou(outputs, gt)
                comet_logger.log_metric("batch_jaccard", jaccard_value.item())
                comet_logger.log_metric("batch_fbiou", fbiou_value.item())
                avg_jaccard.update(jaccard_value.item())
                avg_fbiou.update(fbiou_value.item())
                cur_lr = scheduler.get_lr()[0] if scheduler else train_params["initial_lr"]

            avg_loss.update(loss.item())
            if i == 0:
                first_step_loss.update(loss.item())
                comet_logger.log_metric("first_step_loss", loss.item())
                if tot_steps % comet_logger.log_frequency == 0:
                    first_step_jaccard.update(jaccard_value.item())
                    first_step_fbiou.update(fbiou_value.item())
                    comet_logger.log_metric("first_step_jaccard", jaccard_value.item())
                    comet_logger.log_metric("first_step_fbiou", fbiou_value.item())
                    comet_logger.log_metric("lr", cur_lr)

            comet_logger.log_batch(
                batch_idx=batch_idx,
                image_idx=tot_images,
                batch_size=cur_batch_size,
                epoch=epoch,
                step=tot_steps,
                substitution_step=i,
                input_dict=input_dict,
                gt=gt,
                pred=outputs,
                dataset=dataloader.dataset,
                dataset_names=dataset_names,
                phase="train",
            )
            substitutor.generate_new_points(outputs, gt)
            bar.set_postfix(
                {
                    "loss": loss.item(),
                    "jac/miou": jaccard_value.item(),
                    "fbiou": fbiou_value.item(),
                    "lr": cur_lr,
                }
            )
            tot_steps += 1
        comet_logger.save_experiment_timed()
        tot_images += cur_batch_size

    comet_logger.log_metrics(
        {
            "avg_loss": avg_loss.compute(),
            "avg_jaccard": avg_jaccard.compute(),
            "avg_fbiou": avg_fbiou.compute(),
            "avg_first_step_loss": first_step_loss.compute(),
            "avg_first_step_jaccard": first_step_jaccard.compute(),
            "avg_first_step_fbiou": first_step_fbiou.compute(),
        },
        epoch=epoch,
    )


def validate(model, criterion, dataloader, epoch, comet_logger, accelerator):
    model.eval()
    avg_loss = RunningAverage()
    avg_jaccard = RunningAverage()
    avg_fbiou = RunningAverage()

    dataloader = accelerator.prepare(dataloader)
    dataloader.dataset.log_images = True # Logs the first batch
    tot_steps = 0
    tot_images = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader), postfix={"loss": 0})

    with torch.no_grad():
        for batch_idx, batch_tuple in bar:
            batch_dict, dataset_names = batch_tuple
            batch_dict = next(iter(Substitutor(batch_dict, substitute=False)))
            cur_batch_size = get_batch_size(batch_dict)
            image_dict, gt = batch_dict

            output = model(image_dict)
            jaccard_value = jaccard(output, gt, num_classes=output.shape[1])
            fbiou_value = fbiou(output, gt)
            loss = criterion(output, gt)  # sum up batch loss
            # TODO Log validation images

            avg_loss.update(loss.item())
            avg_jaccard.update(jaccard_value.item())
            avg_fbiou.update(fbiou_value.item())
            bar.set_postfix(
                {
                    "jac/miou": jaccard_value.item(),
                    "fbiou": fbiou_value.item(),
                    "loss": loss.item(),
                }
            )
            comet_logger.log_batch(
                batch_idx=batch_idx,
                image_idx=tot_images,
                batch_size=cur_batch_size,
                epoch=epoch,
                step=tot_steps,
                substitution_step=0,
                input_dict=image_dict,
                gt=gt,
                pred=output,
                dataset=dataloader.dataset,
                dataset_names=dataset_names,
                phase="val",
            )
            tot_steps += 1
            tot_images += cur_batch_size

        comet_logger.log_metrics(
            {
                "jaccard": avg_jaccard.compute(),
                "loss": avg_loss.compute(),
                "fbiou": avg_fbiou.compute(),
            },
            epoch=epoch
        )
    logger.info(f"Validation Epoch {epoch} finished")
    logger.info(f"Jaccard: {avg_jaccard.compute()}")
    logger.info(f"Loss: {avg_loss.compute()}")
    logger.info(f"FBIoU: {avg_fbiou.compute()}")


def test(model, criterion, dataloader, train_dataset, comet_logger):
    model.eval()
    total_loss = 0
    jaccard_value = 0
    fbiou_value = 0

    examples = dataloader.dataset.extract_prompts(
        train_dataset.cat2img,
        train_dataset.img2cat,
        train_dataset.images,
        train_dataset.img2cat_annotations,
    )
    class_embeddings = model.get_class_embeddings(examples)

    with torch.no_grad():
        for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
            image_dict, gt = batch_dict

            output = model.predict(image_dict, class_embeddings)
            jaccard_value += jaccard(output, gt, num_classes=output.shape[1])
            fbiou_value += fbiou(output, gt, num_classes=output.shape[1])
            total_loss += criterion(output, gt).item()  # sum up batch loss

        total_loss /= len(dataloader)
        jaccard_value /= len(dataloader)
        fbiou_value /= len(dataloader)

        comet_logger.log_metrics(
            {"jaccard": jaccard_value, "loss": total_loss, "fbiou": fbiou_value},
        )


def train_and_test(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    comet_logger: Logger,
    train_params,
):
    logger.info("Start training loop...")
    accelerator = Accelerator(even_batches=False)

    criterion = LabelAnythingLoss(train_params["loss"])
    optimizer = AdamW(
        model.get_learnable_params(train_params), lr=train_params["initial_lr"]
    )
    
    scheduler = None
    scheduler_params = train_params.get("scheduler", None)
    if scheduler_params:
        scheduler = get_scheduler(
            scheduler_params["type"],
            optimizer=optimizer,
            num_warmup_steps=scheduler_params["warmup_steps"],
            num_training_steps=train_params["max_epochs"] * len(train_loader),
        )

    if train_params.get("compile", False):
        model = torch.compile(model)

    # Train the Model
    with comet_logger.experiment.train():
        logger.info(f"Running Model Training {args.get('experiment').get('name')}")
        for epoch in range(train_params["max_epochs"]):
            logger.info("Epoch: {}/{}".format(epoch, train_params["max_epochs"]))
            train_epoch(
                model,
                optimizer,
                scheduler,
                criterion,
                train_loader,
                epoch,
                comet_logger,
                accelerator,
                train_params,
            )
            logger.info(f"Finished Epoch {epoch}")

            # save_model(comet_logger.experiment, model, model._get_name())
            if val_loader:
                with comet_logger.experiment.validate():
                    logger.info(f"Running Model Validation")
                    validate(
                        model, criterion, val_loader, epoch, comet_logger, accelerator
                    )
            comet_logger.save_experiment()

    if test_loader:
        with comet_logger.experiment.test():
            logger.info(f"Running Model Testing {args.name}")
            test(model, criterion, test_loader, comet_logger)

    comet_logger.end()
