from typing import Tuple
import numpy as np
import torch

from label_anything import scheduler as schedulers
from label_anything.loss import instiantiate_loss
from label_anything.metrics import metrics_factory


def parse_params(params: dict) -> Tuple[dict, dict, dict, Tuple, dict]:
    # Set Random seeds
    torch.manual_seed(params["train_params"]["seed"])
    np.random.seed(params["train_params"]["seed"])

    # Instantiate loss
    input_train_params = params["train_params"]
    loss_params = params["train_params"]["loss"]
    loss = instiantiate_loss(loss_params["name"], loss_params["params"])

    if "kd" in params:
        if "loss" in params["kd"]:
            loss = init_composed_loss(loss, params["kd"]["loss"])

    if "aux_loss" in params:
        loss = init_composed_loss(loss, params["aux_loss"])

    # metrics
    train_metrics = metrics_factory(params["train_metrics"])
    test_metrics = metrics_factory(params["test_metrics"])

    # dataset params
    dataset_params = params["dataset"]

    train_params = {
        **input_train_params,
        "train_metrics_list": list(train_metrics.values()),
        "valid_metrics_list": list(test_metrics.values()),
        "loss": loss,
        "loss_logging_items_names": ["loss"],
        "sg_logger": params["experiment"]["logger"],
        "sg_logger_params": {
            "entity": params["experiment"]["entity"],
            "tags": params["tags"],
            "project_name": params["experiment"]["name"],
        },
    }

    train_params = parse_scheduler(train_params)

    test_params = {
        "test_metrics": test_metrics,
    }

    # callbacks
    train_callbacks = add_phase_in_callbacks(
        params.get("train_callbacks") or {}, "train"
    )
    test_callbacks = add_phase_in_callbacks(params.get("test_callbacks") or {}, "test")
    val_callbacks = add_phase_in_callbacks(
        params.get("val_callbacks") or {}, "validation"
    )

    # metric to watch
    mwatch = train_params.get("metric_to_watch")
    if mwatch == "loss" or mwatch.split("/")[0] == "loss":
        if hasattr(loss, "component_names"):
            if len(mwatch.split("/")) >= 2:
                mwatch = f'{loss.__class__.__name__}/{"/".join(mwatch.split("/")[1:])}'
            else:
                mwatch = f"{loss.__class__.__name__}/{loss.__class__.__name__}"
        else:
            mwatch = loss.__class__.__name__
        train_params["metric_to_watch"] = mwatch

    # early stopping
    early_stopping = None
    if params.get("early_stopping"):
        early_stopping = params.get("early_stopping")
    elif train_params.get("early_stopping_patience"):
        early_stopping = {"patience": train_params["early_stopping_patience"]}

    if early_stopping:
        val_callbacks["early_stopping"] = early_stopping
        val_callbacks["early_stopping"]["monitor"] = mwatch
        val_callbacks["early_stopping"]["mode"] = (
            "max" if train_params["greater_metric_to_watch_is_better"] else "min"
        )

    # knowledge distillation
    kd = params.get("kd")

    return (
        train_params,
        test_params,
        dataset_params,
        (train_callbacks, val_callbacks, test_callbacks),
        kd,
    )


def init_composed_loss(loss, loss_params):
    """
    Initialize composed loss (e.g. knowledge distillation) and its components
    :param loss: task_loss_function
    :param loss_params: dict of loss parameters
    :return: loss function
    """
    loss_name = loss_params["name"]
    loss_params = loss_params["params"]
    losses_types = [
        loss_type for loss_type in loss_params if loss_type.endswith("_loss")
    ]
    for loss_type in losses_types:
        loss_type_params = loss_params[loss_type].get("params") or {}
        loss_type_name = loss_params.pop(loss_type)[
            "name"
        ]  # remove loss type from loss_params
        loss_params[f"{loss_type}_fn"] = instiantiate_loss(
            loss_type_name, loss_type_params
        )
    return instiantiate_loss(loss_name, {**loss_params, "task_loss_fn": loss})


def add_phase_in_callbacks(callbacks, phase):
    """
    Add default phase to callbacks
    :param callbacks: dict of callbacks
    :param phase: "train", "validation" or "test"
    :return: dict of callbacks with phase
    """
    for callback in callbacks.values():
        if callback.get("phase") is None:
            callback["phase"] = phase
    return callbacks


def parse_scheduler(train_params: dict) -> dict:
    """
    Parse scheduler parameters
    """
    scheduler = train_params.get("scheduler") or train_params.get("lr_mode")
    if scheduler is None:
        return train_params
    if "name" in scheduler:
        params = scheduler.get("params") or {}
        scheduler = scheduler["name"]
    if scheduler in schedulers.__dict__:
        scheduler = schedulers.__dict__[scheduler](**params)
        train_params["lr_mode"] = "function"
        train_params["lr_schedule_function"] = scheduler.perform_scheduling
    return train_params
