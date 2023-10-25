from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from label_anything.data.dataset import LabelAnythingDataset


def get_dataloader(**kwargs):

    preprocess = Compose(
        [
            ToTensor(),
            # Resize((1000, 1000)),
        ]
    )

    dataset = LabelAnythingDataset(
        instances_path="label_anything/data/lvis_v1_train.json",
        preprocess=preprocess,
        num_max_examples=10,
        j_index_value=0.1,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn
    )
    return dataloader