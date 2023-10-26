import torch

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor

from label_anything.data.dataset import LabelAnythingDataset


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
            "embeddings": torch.rand(M, D, H // 4, W // 4),
            "prompt_masks": torch.randint(0, 2, (M, C, H, W)).float(),
            "flags_masks": torch.randint(0, 2, (M, C)),
            "prompt_points": torch.randint(0, 2, (M, C, N, 2)),
            "flags_points": torch.randint(0, 2, (M, C, N)),
            "prompt_bboxes": torch.rand(M, C, N, 4),
            "flags_bboxes": torch.randint(0, 2, (M, C, N)),
            "dims": torch.tensor([H, W]),
            "classes": [{1, 2}, {2, 3}]
        }, torch.randint(0, 2, (M, C, H, W))

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
        return {k: torch.stack(v) if k != "classes" else v for k, v in result_dict.items()}, torch.cat(gt_list)


def get_dataloader(**kwargs):

    preprocess = Compose(
        [
            ToTensor(),
            # Resize((1000, 1000)),
        ]
    )

    # dataset = LabelAnythingDataset(
    #     instances_path="label_anything/data/lvis_v1_train.json",
    #     preprocess=preprocess,
    #     max_num_examples=10,
    #     j_index_value=0.1,
    # )
    dataset = RandomDataset()
    dataloader = DataLoader(
        dataset=dataset, batch_size=kwargs.get("batch_size"), shuffle=False, collate_fn=dataset.collate_fn
    )
    return dataloader