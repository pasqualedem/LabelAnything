import os
import importlib
from inspect import signature
import torch
from ruamel.yaml import YAML, comments
from io import StringIO
import collections.abc
from typing import Mapping
import yaml

from label_anything.models.lam import Lam


def load_yaml(file_path):
    try:
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file.read())
            return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def unwrap_model_from_parallel(model, return_was_wrapped=False):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper
    :param model: the model
    :return: the unwrapped model
    """
    if isinstance(
        model,
        (
            torch.nn.DataParallel,
            torch.nn.parallel.DistributedDataParallel,
            Lam,
        ),
    ):
        if return_was_wrapped:
            return model.module, True
        return model.module
    else:
        if return_was_wrapped:
            return model, False
        return model


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


def log_every_n(image_idx: int, batch_size: int, n: int):
    if n is None:
        return False
    cur_step = image_idx % n
    next_step = (image_idx + batch_size) % n
    return cur_step > next_step


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


def find_divisor_pairs(number):
    divisor_pairs = []
    
    for i in range(1, int(number**0.5) + 1):
        if number % i == 0:
            divisor_pairs.append((i, number // i))
    
    return divisor_pairs


class RunningAverage:
    def __init__(self):
        self.accumulator = 0
        self.steps = 0
        
    def update(self, value):
        self.accumulator += value
        self.steps += 1
        
    def compute(self):
        return self.accumulator / self.steps