import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor

from label_anything.data.dataset import LabelAnythingDataset, VariableBatchSampler
from label_anything.data.coco import CocoLVISTestDataset, CocoLVISDataset
from label_anything.data.dram import DramTestDataset
from label_anything.data.transforms import CustomNormalize, CustomResize
from label_anything.data.weedmap import WeedMapTestDataset


TEST_DATASETS = {
    "test_coco": CocoLVISTestDataset,
    "test_lvis": CocoLVISTestDataset,
    "test_weedmap": WeedMapTestDataset,
    "test_dram": DramTestDataset,
}


def get_dataloaders(dataset_args, dataloader_args, num_processes):
    SIZE = 1024

    preprocess = Compose([CustomResize(SIZE), PILToTensor(), CustomNormalize(SIZE)])
    datasets_params = dataset_args.get("datasets")
    common_params = dataset_args.get("common")
    possible_batch_example_nums = dataloader_args.pop("possible_batch_example_nums")
    val_possible_batch_example_nums = dataloader_args.pop(
        "val_possible_batch_example_nums", possible_batch_example_nums
    )

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

    train_dataset = LabelAnythingDataset(
        datasets_params=train_datasets_params,
        common_params={**common_params, "preprocess": preprocess},
    )
    train_batch_sampler = VariableBatchSampler(
        train_dataset,
        possible_batch_example_nums=possible_batch_example_nums,
        num_processes=num_processes,
        shuffle=True,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        **dataloader_args,
        collate_fn=train_dataset.collate_fn,
        batch_sampler=train_batch_sampler,
    )
    if val_datasets_params:
        val_dataset = LabelAnythingDataset(
            datasets_params=val_datasets_params,
            common_params={**common_params, "preprocess": preprocess},
        )
        val_batch_sampler = VariableBatchSampler(
            val_dataset,
            possible_batch_example_nums=val_possible_batch_example_nums,
            num_processes=num_processes,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            **dataloader_args,
            collate_fn=val_dataset.collate_fn,
            batch_sampler=val_batch_sampler,
        )
    else:
        val_dataloader = None
    if test_datasets_params:
        test_datasets = [
            TEST_DATASETS[dataset](**params, preprocess=preprocess)
            for dataset, params in test_datasets_params.items()
        ]
        test_dataloaders = [
            DataLoader(
                dataset=data,
                **dataloader_args,
                collate_fn=data.collate_fn if hasattr(data, "collate_fn") else None,
            )
            for data in test_datasets
        ]
    else:
        test_dataloaders = None
    return (
        train_dataloader,
        val_dataloader,
        test_dataloaders,
    )
