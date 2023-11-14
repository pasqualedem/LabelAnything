import itertools
import os
import random
import warnings
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
import torchvision.transforms
import label_anything.data.utils as utils
from label_anything.data.examples import ExampleGeneratorPowerLawUniform
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    PILToTensor,
    ToTensor,
)
from label_anything.data.coco import CocoLVISDataset


datasets = {
    "coco": CocoLVISDataset,
    "lvis": CocoLVISDataset,
    "ade20k": None,
}


class LabelAnythingDataset(Dataset):
    def __init__(self, datasets_params, common_params) -> None:
        self.datasets = {
            dataset_name: datasets[dataset_name](**params, **common_params)
            for dataset_name, params in datasets_params.items()
        }
        index = sum([
            [(dataset_name, i) for i in range(len(dataset))]
            for dataset_name, dataset in self.datasets.items()
        ], [])
        self.index = {i: index for i, index in enumerate(index)}
        super().__init__()

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, index) -> Any:
        dataset_name, dataset_index = self.index[index]
        return self.datasets[dataset_name][dataset_index]
