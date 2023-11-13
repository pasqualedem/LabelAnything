import torch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, PILToTensor

from label_anything.data.dataset import LabelAnythingDataset
from label_anything.data.transforms import CustomNormalize, CustomResize


class RandomDataset(Dataset):
    def __init__(self):
        self.len = 100

    def __getitem__(self, index):
        H, W = 1024, 1024
        M = 2
        C = 3
        D = 256
        N = 5
        return {
            "embeddings": torch.rand(M, D, H // 16, W // 16),
            "prompt_masks": torch.randint(0, 2, (M, C, H // 4, W // 4)).float(),
            "flags_masks": torch.randint(0, 2, (M, C)),
            "prompt_points": torch.randint(0, 2, (M, C, N, 2)),
            "flags_points": torch.randint(0, 2, (M, C, N)),
            "prompt_bboxes": torch.rand(M, C, N, 4),
            "flags_bboxes": torch.randint(0, 2, (M, C, N)),
            "dims": torch.tensor([H, W]),
            "classes": [{1, 2}, {2, 3}]
        }, torch.randint(0, 3, (M, H // 4, W // 4))

    def __len__(self):
        return self.len
    
    def collate_fn(self, batch):
        result_dict = {}
        gt_list = []
        for elem in batch:
            dictionary, gts = elem
            gt_list.append(gts)
            for key, value in dictionary.items():
                if key in result_dict:
                    if not isinstance(result_dict[key], list):
                        result_dict[key] = [result_dict[key]]
                    if isinstance(value, list):
                        result_dict[key].extend(value)
                    else:
                        result_dict[key].append(value)
                else:
                    if isinstance(value, list):
                        result_dict[key] = value
                    else:
                        result_dict[key] = [value]
        return {k: torch.stack(v) if k != "classes" else v for k, v in result_dict.items()}, torch.stack(gt_list)


def get_dataloaders(dataset_args, dataloader_args):
    SIZE = 1024

    preprocess = Compose(
        [CustomResize(SIZE), PILToTensor(), CustomNormalize(SIZE)]
    )

    dataset = LabelAnythingDataset(
        **dataset_args,
        preprocess=preprocess,
    )    
    dataloader = DataLoader(
        dataset=dataset, **dataloader_args, collate_fn=dataset.collate_fn
    )
    return dataloader, None, None # placeholder for val and test loaders until Raffaele implements them