import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, PILToTensor, Resize, Normalize, ToTensor

from label_anything.data.dataset import LabelAnythingDataset, VariableBatchSampler
from label_anything.data.coco import CocoLVISTestDataset, CocoLVISDataset
from label_anything.data.transforms import CustomNormalize, CustomResize


TEST_DATASETS = {
    "test_coco": CocoLVISTestDataset,
    "test_lvis": CocoLVISTestDataset,
}
   


def get_dataloaders(dataset_args, dataloader_args, num_processes):
    SIZE = 1024
    size = dataset_args.get("common", {}).get("image_size", SIZE)

    if "custom_preprocess" in dataset_args.get("common", {}):
        custom_preprocess = dataset_args["common"].pop("custom_preprocess")
        mean = custom_preprocess["mean"]
        std = custom_preprocess["std"]
        preprocess = Compose(
            [Resize(size=(size, size)), ToTensor(), Normalize(mean, std)]
        )
    else:
        preprocess = Compose([CustomResize(size), PILToTensor(), CustomNormalize(size)])
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

    test_dataloaders = None
    return (
        train_dataloader,
        val_dataloaders,
        test_dataloaders,
    )
