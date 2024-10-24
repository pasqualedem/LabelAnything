import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tap.data.kvasir import KvarisTestDataset
from tap.data.transforms import Normalize, Resize

from tap.data.dataset import LabelAnythingDataset, VariableBatchSampler
from tap.data.coco import CocoLVISTestDataset, CocoLVISDataset
from tap.data.dram import DramTestDataset, collate_fn as dram_collate
from tap.data.transforms import CustomNormalize, CustomResize
from tap.data.utils import get_mean_std
from tap.data.weedmap import WeedMapTestDataset
from tap.data.brain_mri import BrainMriTestDataset, BrainTestDataset


TEST_DATASETS = {
    "test_coco": CocoLVISTestDataset,
    "test_lvis": CocoLVISTestDataset,
    "test_weedmap": WeedMapTestDataset,
    "test_dram": DramTestDataset,
    "test_brain": BrainTestDataset,
    "test_kvaris": KvarisTestDataset,
}


def map_collate(dataset):
    if isinstance(dataset, DramTestDataset):
        return dram_collate
    return dataset.collate_fn if hasattr(dataset, "collate_fn") else None


def get_preprocessing(params):
    SIZE = 1024
    size = params.get("common", {}).get("image_size", SIZE)
    custom_preprocess = params.get("common", {}).get("custom_preprocess", True)
    if "preprocess" in params.get("common", {}):
        preprocess_params = params["common"].pop("preprocess")
        mean = preprocess_params["mean"]
        std = preprocess_params["std"]
        mean, std = get_mean_std(mean, std)
    else:
        mean, std = get_mean_std("default", "default")
    preprocess = (
        Compose(
            [
                CustomResize(long_side_length=size),
                ToTensor(),
                CustomNormalize(size, mean, std),
            ]
        )
        if custom_preprocess
        else Compose(
            [
                Resize(size=(size, size)),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
    )
    return preprocess


def get_dataloaders(dataset_args, dataloader_args, num_processes):
    preprocess = get_preprocessing(dataset_args)

    datasets_params = dataset_args.get("datasets")
    common_params = dataset_args.get("common")
    possible_batch_example_nums = dataloader_args.pop("possible_batch_example_nums")
    val_possible_batch_example_nums = dataloader_args.pop(
        "val_possible_batch_example_nums", possible_batch_example_nums
    )

    prompt_types = dataloader_args.pop("prompt_types", None)
    prompt_choice_level = dataloader_args.pop("prompt_choice_level", "batch")

    val_prompt_types = dataloader_args.pop("val_prompt_types", prompt_types)
    num_steps = dataloader_args.pop("num_steps", None)

    val_datasets_params = {
        k: v for k, v in datasets_params.items() if k.startswith("val_")
    }
    test_datasets_params = {
        k: v for k, v in datasets_params.items() if k.startswith("test_")
    }
    train_datasets_params = {
        k: v
        for k, v in datasets_params.items()
        if k not in list(val_datasets_params.keys()) + list(test_datasets_params.keys())
    }
    train_dataloader = None
    if train_datasets_params:
        train_dataset = LabelAnythingDataset(
            datasets_params=train_datasets_params,
            common_params={**common_params, "preprocess": preprocess},
        )
        train_batch_sampler = VariableBatchSampler(
            train_dataset,
            possible_batch_example_nums=possible_batch_example_nums,
            num_processes=num_processes,
            prompt_types=prompt_types,
            prompt_choice_level=prompt_choice_level,
            shuffle=True,
            num_steps=num_steps,
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            **dataloader_args,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=train_batch_sampler,
        )
    if val_datasets_params:
        val_dataloaders = {}
        for dataset, params in val_datasets_params.items():
            splits = dataset.split("_")
            if len(splits) > 2:
                dataset_name = "_".join(splits[:2])
            else:
                dataset_name = dataset
            val_dataset = LabelAnythingDataset(
                datasets_params={dataset_name: params},
                common_params={**common_params, "preprocess": preprocess},
            )
            val_batch_sampler = VariableBatchSampler(
                val_dataset,
                possible_batch_example_nums=val_possible_batch_example_nums,
                num_processes=num_processes,
                prompt_types=val_prompt_types,
            )
            val_dataloader = DataLoader(
                dataset=val_dataset,
                **dataloader_args,
                collate_fn=val_dataset.collate_fn,
                batch_sampler=val_batch_sampler,
            )
            val_dataloaders[dataset] = val_dataloader
    else:
        val_dataloaders = None
    if test_datasets_params:
        test_datasets = {
            dataset: TEST_DATASETS[dataset](**params, preprocess=preprocess)
            for dataset, params in test_datasets_params.items()
        }
        test_dataloaders = {
            name: DataLoader(
                dataset=data,
                **dataloader_args,
                collate_fn=map_collate(data),
            )
            for name, data in test_datasets.items()
        }
    else:
        test_dataloaders = None
    return (
        train_dataloader,
        val_dataloaders,
        test_dataloaders,
    )
