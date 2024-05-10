import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, PILToTensor
from typing import Union
from label_anything.data.utils import BatchKeys, collate_gts, flags_merge
from PIL import Image
import random
from label_anything.data.transforms import PromptsProcessor
from label_anything.data.test import LabelAnythingTestDataset


def collate_fn(batched_input):
    images = torch.stack([x[BatchKeys.IMAGES] for x in batched_input])
    dims = torch.stack([x[BatchKeys.DIMS] for x in batched_input])

    max_dims = torch.max(dims, 0).values.tolist()
    gts = torch.stack(
        [collate_gts(x[BatchKeys.GROUND_TRUTHS], max_dims) for x in batched_input]
    )

    return {
        BatchKeys.IMAGES: images.unsqueeze(dim=1),
        BatchKeys.DIMS: dims,
    }, gts


def _get_data(image_dir, split):
    with open(f"{image_dir}/{split}.txt", mode="r") as f:
        data = f.readlines()
    return list(map(lambda x: x.strip(), data))


class DramTestDataset(LabelAnythingTestDataset):
    ID2TRAIN_ID = {
        0: 0,
        3: 1,
        4: 2,
        5: 3,
        8: 4,
        9: 5,
        10: 6,
        12: 7,
        13: 8,
        15: 9,
        16: 10,
        17: 11,
    }
    TRAIN_ID2NAME = {
        0: "background",
        1: "bird",
        2: "boat",
        3: "bottle",
        4: "cat",
        5: "chair",
        6: "cow",
        7: "dog",
        8: "horse",
        9: "person",
        10: "pottedplant",
        11: "sheep",
    }

    HIERARCHY_ID2ID = {
        0: 0,  # BACKGROUND
        1: 1,  # BIRD -> ANIMAL
        2: 2,  # BOAT
        3: 3,  # BOTTLE
        4: 1,  # CAT -> ANIMAL
        5: 4,  # CHAIR
        6: 1,  # COW -> ANIMAL
        7: 1,  # DOG -> ANIMAL
        8: 1,  # HORSE -> ANIMAL
        9: 5,  # PERSON
        10: 6,  # POTTEDPLANT
        11: 1,  # SHEEP -> ANIMAL
    }

    def __init__(
        self,
        image_dir: str,
        gt_dir: str,
        example_image_dir: str,
        example_gt_dir: str,
        split: str = "dram",
        example_split: str = "dram",
        prompt_mask_size: int = 256,
        preprocess: Union[Compose, ToTensor] = ToTensor(),
        custom_preprocess=True,
        hierachy: bool = False,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.split = split
        self.preprocess = preprocess

        self.gt_preprocess = ToTensor()

        self.data = _get_data(self.image_dir, self.split)

        self.example_image_dir = example_image_dir
        self.example_gt_dir = example_gt_dir
        self.example_split = example_split
        self.example_data = _get_data(self.example_image_dir, self.example_split)

        self.hierachy = hierachy
        self.example_img2cat, self.example_cat2img = self._get_support_dict()
        self.prompt_processor = PromptsProcessor(
            masks_side_length=prompt_mask_size, custom_preprocess=custom_preprocess
        )

    def _get_support_dict(self):
        img2cat = {}
        cat2img = {}
        for ix, fname in enumerate(self.example_data):
            gt = self._load_gt(f"{self.example_gt_dir}/{fname}.png")
            classes = torch.unique(gt).tolist()
            img2cat[ix] = classes
            for c in classes:
                if c not in cat2img:
                    cat2img[c] = [ix]
                else:
                    cat2img[c].append(ix)
        return img2cat, cat2img

    def _load_image(self, fname):
        img = Image.open(fname).convert("RGB")
        size = img.size
        if self.preprocess:
            img = self.preprocess(img)
        return img, torch.as_tensor(size[::-1]).t()

    def _load_gt(self, fname):
        gt = Image.open(fname)
        gt = self.gt_preprocess(gt).int()
        copy_gt = torch.zeros_like(gt)
        for k, v in DramTestDataset.ID2TRAIN_ID.items():
            copy_gt[gt == k] = v
            if self.hierachy:
                copy_gt[copy_gt == v] = DramTestDataset.HIERARCHY_ID2ID[v]
        return copy_gt.int().squeeze()

    def __getitem__(self, item):
        fname = self.data[item]
        image, size = self._load_image(f"{self.image_dir}/{fname}.jpg")
        gt = self._load_gt(f"{self.gt_dir}/{fname}.png")
        return {
            BatchKeys.IMAGES: image,
            BatchKeys.DIMS: size,
            BatchKeys.GROUND_TRUTHS: gt,
        }

    def __len__(self):
        return len(self.data)

    def _extract_examples(self):
        prompt_images = set()
        categories = list(DramTestDataset.TRAIN_ID2NAME.keys())
        random.shuffle(categories)
        for cat_id in categories:
            if cat_id not in self.example_cat2img:
                continue
            cat_images = self.example_cat2img[cat_id]
            _, img = max(map(lambda x: (len(self.example_img2cat[x]), x), cat_images))
            prompt_images.add(img)
        return prompt_images

    def _masks_to_tensor(self, masks, cat_ids):
        n = len(masks)
        c = len(cat_ids)
        tensor_shape = (
            n,
            c,
            self.prompt_processor.masks_side_length,
            self.prompt_processor.masks_side_length,
        )
        mask_tensor = torch.zeros(tensor_shape)
        flag_tensor = torch.zeros((n, c))
        for i, annotation in enumerate(masks):
            for j, cat_id in enumerate(annotation):
                mask = self.prompt_processor.apply_masks(annotation[cat_id])
                tensor_mask = torch.tensor(mask)
                mask_tensor[i, j, :] = tensor_mask
                flag_tensor[i, j] = 1 if torch.sum(tensor_mask) > 0 else 0
        return mask_tensor, flag_tensor

    def _get_prompt_masks(self, image_ids, images_fname):
        cat_ids = sorted(list(DramTestDataset.TRAIN_ID2NAME.keys()))
        masks = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        for idx, img in enumerate(image_ids):
            mask = self._load_gt(images_fname[idx])
            for cat_id in cat_ids:
                if cat_id not in self.example_img2cat[img]:
                    continue
                masks[idx][cat_id].append((mask == cat_id).int().numpy())

        # convert
        return self._masks_to_tensor(masks, cat_ids)

    def extract_prompts(self):
        image_ids = self._extract_examples()
        prompt_images_fnames = [self.example_data[x] for x in image_ids]

        images_sizes = [
            self._load_image(f"{self.example_image_dir}/{fname}.jpg")
            for fname in prompt_images_fnames
        ]
        images = torch.stack([x[0] for x in images_sizes])
        sizes = torch.stack([x[1] for x in images_sizes])

        masks_fnames = [f"{self.example_gt_dir}/{x}.png" for x in prompt_images_fnames]
        masks, flag_masks = self._get_prompt_masks(image_ids, masks_fnames)

        # getting flag examples
        flag_examples = flag_masks.clone().bool()

        prompt_dict = {
            BatchKeys.IMAGES: images,
            BatchKeys.FLAG_EXAMPLES: flag_examples[:, 1:],
            BatchKeys.PROMPT_MASKS: masks[:, 1:],
            BatchKeys.FLAG_MASKS: flag_masks[:, 1:],
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict

    @property
    def num_classes(self):
        return len(self.TRAIN_ID2NAME.keys())


if __name__ == "__main__":
    from label_anything.data.transforms import CustomNormalize, CustomResize
    from torch.utils.data import DataLoader

    preprocess = Compose(
        [
            CustomResize(1024),
            PILToTensor(),
            CustomNormalize(),
        ]
    )
    parent_dir = "data/raw/DRAM_processed"
    dram = DramTestDataset(
        image_dir=f"{parent_dir}/test",
        gt_dir=f"{parent_dir}/labels",
        preprocess=preprocess,
        example_image_dir=f"{parent_dir}/test",
        example_gt_dir=f"{parent_dir}/labels",
        example_split="dram",
    )

    print("Prompts fetching test")
    prompts = dram.extract_prompts()
    for k, v in prompts.items():
        print(f"{k}: {v.size()}")
    print("\n\n")

    print("Single example fetching test")
    data_point = dram[0]
    for k, v in data_point.items():
        print(f"{k}: {v.size()}")
    print("\n\n")

    print("Loader fetching test")
    loader = DataLoader(dram, collate_fn=collate_fn, batch_size=8)
    for data_point, gt in loader:
        for k, v in data_point.items():
            print(f"{k}: {v.size()}")
        print(f"gt: {gt.size()}")
        exit()
