import os
import glob
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def download_and_extract_dataset(
    username="mateuszbuda",
    dataset_name="lgg-mri-segmentation",
    path="data/raw",
):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(f"{username}/{dataset_name}", unzip=True, path=path)
    print("Downloaded and extracted dataset:", dataset_name)


def generate_data_map(path):
    data_map = []
    for sub_dir_path in glob.glob(path + "*"):
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dirname, image_path])
    return data_map


def generate_df(path):
    data_map = generate_data_map(path)
    df = pd.DataFrame({"dirname": data_map[::2], "path": data_map[1::2]})
    BASE_LEN = len(os.path.join(path, "TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_"))
    END_IMG_LEN = 4
    END_MASK_LEN = 9

    df_imgs = df[~df["path"].str.contains("mask")]
    df_masks = df[df["path"].str.contains("mask")]

    imgs = sorted(df_imgs["path"].values, key=lambda x: int(x[BASE_LEN:-END_IMG_LEN]))
    masks = sorted(
        df_masks["path"].values, key=lambda x: int(x[BASE_LEN:-END_MASK_LEN])
    )

    df = pd.DataFrame(
        {"patient": df_imgs.dirname.values, "image_path": imgs, "mask_path": masks}
    )

    df["diagnosis"] = df["mask_path"].apply(lambda m: diagnose(m))

    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    try:
        df.to_csv("data/processed/brain_mri_df.csv", index=False)
    except Exception as e:
        print(e)


def diagnose(mask_path):
    value = np.max(cv2.imread(mask_path))
    return 1 if value > 0 else 0


class BrainMriDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = pd.read_csv(df, index_col=False)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 2], 0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        data_dict = {
            "image": image,
            "mask": mask,
        }

        return data_dict


if __name__ == "__main__":
    if os.path.exists("data/raw/lgg-mri-segmentation"):
        print("Dataset already downloaded and extracted")
    else:
        download_and_extract_dataset()  # all parameters are set by default

    path = "data/raw/lgg-mri-segmentation/kaggle_3m/"
    generate_df(path)
    dataset = BrainMriDataset(
        "data/processed/brain_mri_df.csv",
        transforms=transforms.ToTensor(),
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    i = next(iter(dataloader))
    print(f'image: {i["image"].shape}, mask: {i["mask"].shape}')
    # image: torch.Size([4, 3, 256, 256]), mask: torch.Size([4, 1, 256, 256])
