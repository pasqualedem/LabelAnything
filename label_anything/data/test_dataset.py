from torchvision.transforms import ToTensor
from dataset import LabelAnythingDataset
import utils
from typing import Any, Dict, List, Tuple
import torch


class LabelAnythingTestDataset(LabelAnythingDataset):
    def __init__(
            self,
            instances_path,  # Path
            img_dir=None,  # directory (only if images have to be loaded from disk)
            max_num_examples=10,  # number of max examples to be given for the target image
            preprocess=ToTensor(),  # preprocess step
            j_index_value=0.5,  # threshold for extracting examples
            seed=42,  # for reproducibility
            max_mum_coords=10,  # max number of coords for each example for each class
    ):
        super(LabelAnythingTestDataset, self).__init__(instances_path, img_dir, max_num_examples, preprocess,
                                                       j_index_value, seed, max_mum_coords)

    def __getitem__(self, item):
        base_image_data = self.images[self.image_ids[item]]
        image = self._load_image(base_image_data)
        if self.preprocess:
            image = self.preprocess(image)
        img_id = base_image_data['id']
        gt = self.get_ground_truths([img_id], self.img2cat[img_id])
        gt = torch.tensor(gt[0])
        dim = torch.as_tensor(gt.size())
        data_dict = {
            "image": image,
            "dim": dim,
            "gt": gt,
        }
        return data_dict

    def collate_fn(
        self, batched_input: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        images = torch.stack([x["image"] for x in batched_input])

        dims = torch.stack([x["dim"] for x in batched_input])

        max_dims = torch.max(dims, 0).values.tolist()
        gt = torch.stack([utils.collate_gts(x["gt"], max_dims) for x in batched_input])

        data_dict = {
            "images": images,
            "dims": dims,
        }

        return data_dict, gt


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, PILToTensor
    from transforms import CustomNormalize, CustomResize
    import torch

    preprocess = Compose(
        [
            CustomResize(1024),
            PILToTensor(),
            CustomNormalize(),
        ]
    )

    dataset = LabelAnythingTestDataset(
        instances_path="lvis_v1_train.json",
        preprocess=preprocess,
        max_num_examples=10,
        j_index_value=0.1,
    )

    """x = dataset[1]
    print([f'{k}: {v.size()}' for k, v in x.items() if isinstance(v, torch.Tensor)])
    exit()"""

    dataloader = DataLoader(
        dataset=dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn
    )
    data_dict, gt = next(iter(dataloader))

    print([f"{k}: {v.size() if isinstance(v, torch.Tensor) else v}" for k, v in data_dict.items()])
    print(f"gt: {gt.size()}")


