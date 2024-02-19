import os
import importlib
from datetime import datetime
from inspect import signature
import time
import torch
from ruamel.yaml import YAML, comments
from io import StringIO
import collections.abc
from typing import Mapping
import yaml
from label_anything.data.utils import StrEnum

# from label_anything.models.lam import Lam 


FLOAT_PRECISIONS = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def strip_wandb_keys_recursive(data):
    
    if isinstance(data, dict):
        d = {}
        for k, v in data.items():
            if k == "wandb_version":
                continue
            elif k == "_wandb":
                d = {**d, **strip_wandb_keys_recursive(v)}
            elif k == "desc":
                continue
            elif k == "value":
                d = {**d, **strip_wandb_keys_recursive(v)}
            else:
                d[k] = strip_wandb_keys_recursive(v)
        return d
    elif isinstance(data, list):
        return [strip_wandb_keys_recursive(v) for v in data]
    else:
        return data
    

def strip_wandb_keys(data):
    if "_wandb" in data:
        return strip_wandb_keys_recursive(data)
    return data


def load_yaml(file_path):
    try:
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file.read())
            data = strip_wandb_keys(data)
            return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        

def write_yaml(data: dict, file_path: str = None, file=None):
    """ Write a dictionary to a YAML file.

    Args:
        data (dict): the data to write
        file_path (str): the path to the file
        file: the file object to write to (esclusive with file_path)
    """
    if file is not None:
        file.write(yaml.dump(data))
        return
    if file_path is None:
        raise ValueError("file_path or file must be specified")
    try:
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


# def unwrap_model_from_parallel(model, return_was_wrapped=False):
#     """
#     Unwrap a model from a DataParallel or DistributedDataParallel wrapper
#     :param model: the model
#     :return: the unwrapped model
#     """
#     if isinstance(
#         model,
#         (
#             torch.nn.DataParallel,
#             torch.nn.parallel.DistributedDataParallel,
#             Lam,
#         ),
#     ):
#         if return_was_wrapped:
#             return model.module, True
#         return model.module
#     else:
#         if return_was_wrapped:
#             return model, False
#         return model


def get_module_class_from_path(path):
    path = os.path.normpath(path)
    splitted = path.split(os.sep)
    module = ".".join(splitted[:-1])
    cls = splitted[-1]
    return module, cls


def update_collection(collec, value, key=None):
    if isinstance(collec, dict):
        if isinstance(value, dict):
            for keyv, valuev in value.items():
                collec = update_collection(collec, valuev, keyv)
        elif key is not None:
            if value is not None:
                collec[key] = value
        else:
            collec = {**collec, **value} if value is not None else collec
    else:
        collec = value if value is not None else collec
    return collec


def nested_dict_update(d, u):
    if u is not None:
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = nested_dict_update(d.get(k) or {}, v)
            else:
                d[k] = v
    return d


def instantiate_class(name, params):
    module, cls = get_module_class_from_path(name)
    imp_module = importlib.import_module(module)
    imp_cls = getattr(imp_module, cls)
    if (
        len(signature(imp_cls).parameters.keys()) == 1
        and "params" in list(signature(imp_cls).parameters.keys())[0]
    ):
        return imp_cls(params)
    return imp_cls(**params)


def substitute_values(x: torch.Tensor, values, unique=None):
    """
    Substitute values in a tensor with the given values
    :param x: the tensor
    :param unique: the unique values to substitute
    :param values: the values to substitute with
    :return: the tensor with the values substituted
    """
    if unique is None:
        unique = x.unique()
    lt = torch.full((unique.max() + 1,), -1, dtype=values.dtype, device=x.device)
    lt[unique] = values
    return lt[x]


# def load_yaml(path, return_string=False):
#     if hasattr(path, "readlines"):
#         d = convert_commentedmap_to_dict(YAML().load(path))
#         if return_string:
#             path.seek(0)
#             return d, path.read().decode("utf-8")
#     with open(path, "r") as param_stream:
#         d = convert_commentedmap_to_dict(YAML().load(param_stream))
#         if return_string:
#             param_stream.seek(0)
#             return d, str(param_stream.read())
#     return d


def convert_commentedmap_to_dict(data):
    """
    Recursive function to convert CommentedMap to dict
    """
    if isinstance(data, comments.CommentedMap):
        result = {}
        for key, value in data.items():
            result[key] = convert_commentedmap_to_dict(value)
        return result
    elif isinstance(data, list):
        return [convert_commentedmap_to_dict(item) for item in data]
    else:
        return data


def log_every_n(image_idx: int, n: int):
    if n is None:
        return False
    return image_idx % n == 0


def dict_to_yaml_string(mapping: Mapping) -> str:
    """
    Convert a nested dictionary or list to a string
    """
    string_stream = StringIO()
    yaml = YAML()
    yaml.dump(mapping, string_stream)
    output_str = string_stream.getvalue()
    string_stream.close()
    return output_str


def get_checkpoints_dir_path(
    project_name: str, group_name: str, ckpt_root_dir: str = None
):
    """Creating the checkpoint directory of a given experiment.
    :param experiment_name:     Name of the experiment.
    :param ckpt_root_dir:       Local root directory path where all experiment logging directories will
                                reside. When none is give, it is assumed that pkg_resources.resource_filename('checkpoints', "")
                                exists and will be used.
    :return:                    checkpoints_dir_path
    """
    if ckpt_root_dir:
        return os.path.join(ckpt_root_dir, project_name, group_name)


def get_timestamp():
    # Get the current timestamp
    timestamp = time.time()  # replace this with your timestamp or use time.time() for current time

    # Convert timestamp to a datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Format the datetime object as a folder-friendly string
    return dt_object.strftime("%Y%m%d_%H%M%S")

def find_divisor_pairs(number):
    divisor_pairs = []
    
    for i in range(1, int(number**0.5) + 1):
        if number % i == 0:
            divisor_pairs.append((i, number // i))
    
    return divisor_pairs


def get_divisors(n):
    """
    Returns a list of divisors of a given number.

    Args:
        n (int): The number to find divisors for.

    Returns:
        list: A list of divisors of the given number.
    """
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


class RunningAverage:
    def __init__(self):
        self.accumulator = 0
        self.steps = 0
        
    def update(self, value):
        self.accumulator += value
        self.steps += 1
        
    def compute(self):
        return self.accumulator / self.steps


class ResultDict(StrEnum):
    CLASS_EMBS = "class_embeddings"
    LOGITS = "logits"
    EXAMPLES_CLASS_EMBS = "class_examples_embeddings"
    LOSS = "loss"