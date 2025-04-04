from enum import Enum
import gc
import os
import contextlib
from copy import deepcopy

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import get_scheduler as get_transformers_scheduler
from torch.optim import AdamW, SGD

from label_anything.data.utils import BatchKeys, random_batch
from label_anything.logger.text_logger import get_logger
from label_anything.utils.utils import find_divisor_pairs, get_divisors, torch_dict_load

logger = get_logger(__name__)


def parse_params(params_dict):
    train_params = params_dict.get("train_params", None)
    val_params = params_dict.get("val_params", {})
    dataset_params = params_dict.get("dataset", {})
    model_params = params_dict.get("model", {})
    prompt_encoder_params = params_dict.get("prompt_encoder", {})
    dataloader_params = params_dict.get("dataloader", {})

    return (
        train_params,
        val_params,
        dataset_params,
        dataloader_params,
        model_params,
        prompt_encoder_params,
    )


def cast_model(model: torch.nn.Module, precision=torch.float32):
    if precision == torch.float32:
        return model
    model = model.type(precision)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.float()
    return model


class SchedulerStepMoment(Enum):
    BATCH = "batch"
    EPOCH = "epoch"


def get_optimizer(parameters, optimizer_params, initial_lr):
    optimizers = {
        "AdamW": AdamW,
        "SGD": SGD,
    }
    if isinstance(optimizer_params, str):
        optimizer_type = optimizer_params
        optimizer_params = {}
    else:
        optimizer_type = optimizer_params.get("type")
    if optimizer_type not in optimizers:
        logger.warning(f"Unknown optimizer type {optimizer_type}, using AdamW")
        optimizer_type = "AdamW"
    logger.info(f"Using optimizer {optimizer_type}")
    optimizer_type = optimizers[optimizer_type]
    return optimizer_type(
        parameters,
        **{
            **{k: v for k, v in optimizer_params.items() if k != "type"},
            "lr": initial_lr,
        },
    )


def get_scheduler(optimizer, num_training_steps, scheduler_params):
    scheduler_params = deepcopy(scheduler_params)
    scheduler_type = scheduler_params.pop("type")
    if scheduler_type is None:
        logger.warning("No scheduler type specified, using None")
        return None, None
    step_moment = scheduler_params.pop("step_moment", None)
    if step_moment is None:
        raise ValueError("step_moment must be specified, choose (batch, epoch)")
    step_moment = SchedulerStepMoment(step_moment)
    num_warmup_steps = scheduler_params.pop("num_warmup_steps", None)
    if scheduler_params:
        logger.warning(
            f"Unused scheduler parameters: {', '.join(list(scheduler_params.keys()))}"
        )
    return (
        get_transformers_scheduler(
            scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        ),
        step_moment,
    )


def get_batch_size(batch_tuple):
    if batch_tuple[0].get("images") is not None:
        return batch_tuple[0]["images"].shape[0]
    if batch_tuple[0].get("embeddings") is not None:
        if isinstance(batch_tuple[0]["embeddings"], dict):
            # get the first key of the dict
            key = list(batch_tuple[0]["embeddings"].keys())[0]
            return batch_tuple[0]["embeddings"][key].shape[0]
        return batch_tuple[0]["embeddings"].shape[0]


def compose_loss_input(input_dict: dict, result_dict: dict):
    return {
        **result_dict,
        BatchKeys.FLAG_EXAMPLES: input_dict[BatchKeys.FLAG_EXAMPLES],
    }


def get_example_class_size(batch_input):
    if batch_input.get("prompt_points") is not None:
        return (
            batch_input["prompt_points"].shape[1],
            batch_input["prompt_points"].shape[2],
        )
    if batch_input.get("prompt_bboxes") is not None:
        return (
            batch_input["prompt_bboxes"].shape[1],
            batch_input["prompt_bboxes"].shape[2],
        )
    if batch_input.get("prompt_masks") is not None:
        return (
            batch_input["prompt_masks"].shape[1],
            batch_input["prompt_masks"].shape[2],
        )


def check_nan(model, input_dict, output, gt, loss, step, train_params):
    if not train_params.get("check_nan", False):
        return
    if step % train_params["check_nan"] != 0:
        return
    if torch.isnan(loss) or loss.detach() in [torch.inf, -torch.inf]:
        if (
            train_params["check_nan"] == 1
        ):  # Makes sense only if we are checking every step
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


def set_class_embeddings(
    accelerator,
    model,
    examples,
):
    examples = {
        k: v.unsqueeze(dim=0).to(accelerator.device) for k, v in examples.items()
    }
    example_size, num_classes = get_example_class_size(examples)
    chunk_sizes = [None] + list(reversed(get_divisors(example_size * num_classes)))
    chunk_sizes = [1]
    passed = False
    i = 0
    while not passed and i < len(chunk_sizes):
        try:
            with torch.no_grad():
                class_embeddings = model.generate_class_embeddings(
                    examples, chunk_size=chunk_sizes[i]
                )
            passed = True
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise e
            gc.collect()
            torch.cuda.empty_cache()
            logger.warning(
                f"Out of memory while generating class embeddings with chunk size {chunk_sizes[i]}"
            )
            exc = e
        i += 1
    if not passed:
        logger.error(
            "Out of memory while generating class embeddings, raising exception"
        )
        raise exc
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.class_embeddings = class_embeddings
    else:
        model.class_embeddings = class_embeddings
    return model


@contextlib.contextmanager
def nosync_accumulation(accumulate=False, accelerator=None, model=None):
    if accumulate:
        with accelerator.no_sync(model):
            yield
    else:
        with contextlib.nullcontext():
            yield


def raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


class WrapperModule(torch.nn.Module):
    def __init__(self, model, loss) -> None:
        super().__init__()
        self.model = model
        self.loss = loss

        self.predict = (
            self.model.predict
            if hasattr(self.model, "predict")
            else raise_not_implemented_error
        )
        self.generate_class_embeddings = (
            self.model.generate_class_embeddings
            if hasattr(self.model, "generate_class_embeddings")
            else raise_not_implemented_error
        )

    def forward(self, input_dict, gt):
        result_dict = self.model(input_dict)
        if self.loss is None:
            return result_dict
        loss = self.loss(compose_loss_input(input_dict, result_dict), gt)
        return {"loss": loss, **result_dict}

    def get_learnable_params(self, train_params):
        model_params = list(self.model.get_learnable_params(train_params))
        loss_params = list(self.loss.parameters())
        if isinstance(model_params[0], dict):
            loss_params = [{"params": loss_params}]
        return model_params + loss_params

    @property
    def class_embeddings(self):
        return self.model.class_embeddings

    @class_embeddings.setter
    def class_embeddings(self, value):
        self.model.class_embeddings = value


def convert_no_vit_checkpoint(model, no_vit_state_dict):
    """
    Convert a checkpoint from a model without Vision Transformer to a model with Vision Transformer

    Args:
        model: Label Anything model with Vision Transformer
        no_vit_state_dict: Checkpoint from a model without Vision Transformer
    """
    state_dict = torch_dict_load(no_vit_state_dict)

    state_dict = {
        **{
            f"model.image_encoder.{k}": v
            for k, v in model.image_encoder.state_dict().items()
        },
        **state_dict,
    }
    return state_dict
