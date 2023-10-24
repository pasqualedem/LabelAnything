from typing import Tuple
import numpy as np
import torch


from label_anything import scheduler as schedulers
from label_anything.metrics import metrics_factory


def parse_params(params_dict):
    train_params = params_dict.get("parameters", {}).get("train_params", {})
    dataset_params = params_dict.get("parameters", {}).get("dataset", {})
    model_params = params_dict.get("parameters", {}).get("model", {})

    return train_params, dataset_params, model_params


# def parse_params(params: dict) -> Tuple[dict, dict, dict, Tuple, dict]:
#     # Set Random seeds
#     torch.manual_seed(params["train_params"]["seed"])
#     np.random.seed(params["train_params"]["seed"])

#     # Instantiate loss
#     input_train_params = params["train_params"]
#     loss_params = params["train_params"]["loss"]
#     loss = instiantiate_loss(loss_params["name"], loss_params["params"])

#     # metrics
#     train_metrics = metrics_factory(params["train_metrics"])
#     test_metrics = metrics_factory(params["test_metrics"])

#     # dataset params
#     dataset_params = params["dataset"]

#     train_params = {
#         **input_train_params,
#         "train_metrics": train_metrics,
#         "valid_metrics": test_metrics,
#         "loss": loss,
#         "loss_logging_items_names": ["loss"],
#         "sg_logger": params["experiment"]["logger"],
#         "sg_logger_params": {
#             "entity": params["experiment"]["entity"],
#             "tags": params["tags"],
#             "project_name": params["experiment"]["name"],
#         },
#     }

#     train_params = parse_scheduler(train_params)
#     model_params = params["model"]

#     return (
#         train_params,
#         dataset_params,
#         model_params
#     )


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
    if scheduler in LR_SCHEDULERS_CLS_DICT:
        return train_params
    if scheduler in schedulers.__dict__:
        scheduler = schedulers.__dict__[scheduler](**params)
        train_params["lr_mode"] = "function"
        train_params["lr_schedule_function"] = scheduler.perform_scheduling
    return train_params
