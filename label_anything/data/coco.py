import itertools
import os
import random
import warnings
from enum import IntEnum
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
import torchvision.transforms
from PIL import Image
from safetensors import safe_open
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, ToTensor

import label_anything.data.utils as utils
from label_anything.data.examples import (
    ExampleGeneratorPowerLawUniform,
    uniform_sampling,
)
from label_anything.data.transforms import (
    CustomNormalize,
    CustomResize,
    PromptsProcessor,
)

warnings.filterwarnings("ignore")


class PromptType(Enum):
    BBOX = 0
    MASK = 1
    POINT = 2


class Label(IntEnum):
    POSITIVE = 1
    NULL = 0
    NEGATIVE = -1


class CocoLVISDataset(Dataset):
    def __init__(
        self,
        instances_path,  # Path
        img_dir=None,  # directory (only if images have to be loaded from disk)
        max_num_examples=10,  # number of max examples to be given for the target image
        preprocess=ToTensor(),  # preprocess step
        seed=42,  # for reproducibility
        emb_dir=None,
        n_folds=-1,
        val_fold=-1,
        load_embeddings=False,
        split="train",
        do_subsample=True,
        add_box_noise=True
    ):
        super().__init__()
        print(f"Loading dataset annotations from {instances_path}...")
        instances = utils.load_instances(instances_path)
        self.emb_dir = emb_dir
        self.load_embeddings = load_embeddings
        self.load_from_dir = img_dir is not None
        self.img_dir = img_dir
        self.log_images = False

        # id to annotation
        self.annotations = {x["id"]: x for x in instances["annotations"]}
        # id to category
        self.categories = {x["id"]: x for x in instances["categories"]}

        # to use with FSS benchmarks
        self.n_folds = n_folds
        self.val_fold = val_fold
        self.split = split
        if self.val_fold != -1:
            assert self.n_folds > 0
            self.__prepare_benchmark()

        # useful dicts
        (
            self.img2cat,
            self.img2cat_annotations,
            self.cat2img,
            self.cat2img_annotations,
        ) = self.__load_annotation_dicts()

        # list of image ids for __getitem__
        img2cat_keys = set(self.img2cat.keys())
        self.image_ids = [
            x["id"] for x in instances["images"] if x["id"] in img2cat_keys
        ]
        self.images = {
            x["id"]: x for x in instances["images"] if x["id"] in img2cat_keys
        }

        # example generator/selector
        self.example_generator = ExampleGeneratorPowerLawUniform(
            categories_to_imgs=self.cat2img
        )

        # max number of examples for each image
        self.max_num_examples = max_num_examples

        # assert that they are positive
        assert self.max_num_examples > 0

        # image preprocessing
        self.preprocess = preprocess
        # prompt preprocessing
        self.prompts_processor = PromptsProcessor(
            long_side_length=1024, masks_side_length=256
        )

        self.seed = seed
        self.do_subsample = do_subsample
        self.add_box_noise = add_box_noise
        self.num_examples = None
        self.__set_all_seeds()

    def __set_all_seeds(self):
        """Enable reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.log_images = True

    def __prepare_benchmark(self):
        """Prepare the dataset for benchmark training."""
        n_categories = len(self.categories)
        n_val_categories = n_categories // self.n_folds
        val_categories_idxs = set(
            self.val_fold + self.n_folds * v for v in range(n_val_categories)
        )
        train_categories_idxs = (
            set(x for x in range(n_categories)) - val_categories_idxs
        )
        categories_idxs = (
            val_categories_idxs if self.split == "val" else train_categories_idxs
        )
        self.categories = {
            k: v
            for i, (k, v) in enumerate(self.categories.items())
            if i in categories_idxs
        }
        category_ids = set(self.categories.keys())
        self.annotations = {
            k: v
            for k, v in self.annotations.items()
            if v["category_id"] in category_ids
        }

    def __load_annotation_dicts(self) -> (dict, dict):
        """Prepares dictionaries for fast access to annotations.

        Returns:
            (dict, dict): Returns four dictionaries:
                1. img2cat: A dictionary mapping image ids to the set of category ids of the annotations of that image.
                2. img2cat_annotations: A dictionary mapping image ids to the annotations of that image.
                3. cat2img: A dictionary mapping category ids to the set of image ids of the annotations of that category.
                4. cat2img_annotations: A dictionary mapping category ids to the annotations of that category.
        """
        img2cat_annotations = {}
        cat2img_annotations = {}

        img2cat = {}
        cat2img = {}

        for ann in self.annotations.values():
            if "iscrowd" in ann and ann["iscrowd"] == 1:
                continue
            if ann["image_id"] not in img2cat_annotations:
                img2cat_annotations[ann["image_id"]] = {}
                img2cat[ann["image_id"]] = set()

            if ann["category_id"] not in img2cat_annotations[ann["image_id"]]:
                img2cat_annotations[ann["image_id"]][ann["category_id"]] = []
                img2cat[ann["image_id"]].add(ann["category_id"])

            img2cat_annotations[ann["image_id"]][ann["category_id"]].append(ann)

            if ann["category_id"] not in cat2img_annotations:
                cat2img_annotations[ann["category_id"]] = {}
                cat2img[ann["category_id"]] = set()

            if ann["image_id"] not in cat2img_annotations[ann["category_id"]]:
                cat2img_annotations[ann["category_id"]][ann["image_id"]] = []
                cat2img[ann["category_id"]].add(ann["image_id"])

            cat2img_annotations[ann["category_id"]][ann["image_id"]].append(ann)

        return img2cat, img2cat_annotations, cat2img, cat2img_annotations

    def __load_safe_embeddings(self, img_data):
        with safe_open(
            f"{self.emb_dir}/{str(img_data['id']).zfill(12)}.safetensors", framework="pt"
        ) as f:
            tensor = f.get_tensor("embedding")
        return tensor

    def _load_image(self, img_data: dict) -> Image:
        """Load an image from disk or from url.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            PIL.Image: The loaded image.
        """
        if self.load_from_dir:
            return Image.open(
                f'{self.img_dir}/{img_data["coco_url"].split("/")[-1]}'
            ).convert("RGB")
        return Image.open(BytesIO(requests.get(img_data["coco_url"]).content)).convert(
            "RGB"
        )

    def _extract_examples(self, img_data: dict) -> (list, list):
        """Chooses examples (and categories) for the query image.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            (list, list): Returns two lists:
                1. examples: A list of image ids of the examples.
                2. cats: A list of sets of category ids of the examples.
        """
        img_cats = torch.tensor(list(self.img2cat[img_data["id"]]))
        sampled_classes = (
            self.example_generator.sample_classes_from_query(
                img_cats, self.example_generator.uniform_sampling
            )
            if self.do_subsample
            else img_cats
        )
        return self.example_generator.generate_examples(
            query_image_id=img_data["id"],
            sampled_classes=torch.tensor(sampled_classes),
            num_examples=self.num_examples,
        )

    def _get_annotations(self, image_ids, cat_ids):
        bboxes = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}
        masks = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}
        points = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}

        # get prompts from annotations
        classes = {img_id: list() for img_id in image_ids}

        for img_id in image_ids:
            img_size = (self.images[img_id]["height"], self.images[img_id]["width"])
            for cat_id in cat_ids:
                # for each pair (image img_id and category cat_id)
                if cat_id not in self.img2cat_annotations[img_id]:
                    # the chosen category is not in the iamge
                    continue

                classes[img_id].append(cat_id)
                for ann in self.img2cat_annotations[img_id][cat_id]:
                    # choose the prompt type
                    prompt_type = random.choice(list(PromptType))

                    if prompt_type == PromptType.BBOX:
                        # take the bbox
                        bboxes[img_id][cat_id].append(
                            self.prompts_processor.convert_bbox(
                                ann["bbox"],
                                *img_size,
                                noise=self.add_box_noise,
                            ),
                        )
                    elif prompt_type == PromptType.MASK:
                        # take the mask
                        masks[img_id][cat_id].append(
                            self.prompts_processor.convert_mask(
                                ann["segmentation"],
                                *img_size,
                            )
                        )
                    elif prompt_type == PromptType.POINT:
                        # take the point
                        mask = self.prompts_processor.convert_mask(
                            ann["segmentation"],
                            *img_size,
                        )
                        points[img_id][cat_id].append(
                            self.prompts_processor.sample_point(mask)
                        )

        # convert the lists of prompts to arrays
        for img_id in image_ids:
            for cat_id in cat_ids:
                bboxes[img_id][cat_id] = np.array((bboxes[img_id][cat_id]))
                masks[img_id][cat_id] = np.array((masks[img_id][cat_id]))
                points[img_id][cat_id] = np.array((points[img_id][cat_id]))
        return bboxes, masks, points, classes

    def _load_and_preprocess_image(self, image_data):
        image = self._load_image(image_data)
        return image if not self.preprocess else self.preprocess(image)

    def _get_images_or_embeddings(self, image_ids):
        if self.load_embeddings:
            images = [
                self.__load_safe_embeddings(image_data)
                for image_data in [self.images[image_id] for image_id in image_ids]
            ]
            return torch.stack(images), "embeddings"
        images = [
            self._load_and_preprocess_image(image_data)
            for image_data in [self.images[image_id] for image_id in image_ids]
        ]
        return images, "images"

    def __getitem__(self, item: int) -> dict:
        if self.split == "train":
            base_image_data = self.images[self.image_ids[item]]
            image_ids, aux_cat_ids = self._extract_examples(base_image_data)
            cat_ids = list(set(itertools.chain(*aux_cat_ids)))
            cat_ids.insert(0, -1)  # add the background class
        else:
            # take a random category (use numpy)
            cat_id = np.random.choice(list(self.categories.keys()))
            # take two random images from that category
            image_ids = np.random.choice(
                list(self.cat2img_annotations[cat_id].keys()), 2, replace=False
            )
            cat_ids = [-1, cat_id]

        # load, stack and preprocess the images
        images, image_key = self._get_images_or_embeddings(image_ids)

        # create the prompt dicts
        bboxes, masks, points, classes = self._get_annotations(image_ids, cat_ids)

        # obtain padded tensors
        bboxes, flag_bboxes = self.annotations_to_tensor(bboxes, PromptType.BBOX)
        masks, flag_masks = self.annotations_to_tensor(masks, PromptType.MASK)
        points, flag_points = self.annotations_to_tensor(points, PromptType.POINT)

        # obtain ground truths
        ground_truths = self.get_ground_truths(image_ids, cat_ids)
        dims = torch.tensor(list(map(lambda x: x.size(), ground_truths)))
        max_dims = torch.max(dims, 0).values.tolist()
        ground_truths = torch.stack(
            [utils.collate_gts(x, max_dims) for x in ground_truths]
        )

        data_dict = {
            image_key: images,
            "prompt_masks": masks,
            "flag_masks": flag_masks,
            "prompt_points": points,
            "flag_points": flag_points,
            "prompt_bboxes": bboxes,
            "flag_bboxes": flag_bboxes,
            "dims": dims,
            "classes": list(classes.values()),
            "ground_truths": ground_truths,
        }

        if self.log_images and self.load_embeddings:
            log_images = [
                self._load_and_preprocess_image(image_data)
                for image_data in [self.images[image_id] for image_id in image_ids]
            ]
            data_dict["images"] = torch.stack(log_images)

        return data_dict

    def get_ground_truths(self, image_ids, cat_ids):
        # initialization
        ground_truths = dict((img_id, {}) for img_id in image_ids)
        # generate masks
        for img_id in image_ids:
            img_size = (self.images[img_id]["height"], self.images[img_id]["width"])
            for cat_id in cat_ids:
                ground_truths[img_id][cat_id] = np.zeros(img_size, dtype=np.int64)
                # zero mask for no segmentation
                if cat_id not in self.img2cat_annotations[img_id]:
                    continue
                for ann in self.img2cat_annotations[img_id][cat_id]:
                    ground_truths[img_id][cat_id] = np.logical_or(
                        ground_truths[img_id][cat_id],
                        self.prompts_processor.convert_mask(
                            ann["segmentation"], *img_size
                        ),
                    )
            # make the ground truth tensor for image img_id
            ground_truth = torch.from_numpy(
                np.array(
                    [
                        ground_truths[img_id][cat_id].astype(np.int64)
                        for cat_id in cat_ids
                    ]
                )
            )
            ground_truths[img_id] = torch.argmax(ground_truth, 0)

        return list(ground_truths.values())

    def __len__(self):
        return len(self.images) if self.split == "train" else 1000

    def annotations_to_tensor(self, annotations, prompt_type) -> torch.Tensor:
        """Transform a dict of annotations of prompt_type to a padded tensor.

        Args:
            annotations (dict): annotations (dict of dicts with np.ndarray as values)
            prompt_type (PromptType): prompt type

        Returns:
            torch.Tensor: padded tensor
        """
        n = len(annotations)
        c = len(next(iter(annotations.values())))

        if prompt_type == PromptType.BBOX:
            max_annotations = utils.get_max_annotations(annotations)
            tensor_shape = (n, c, max_annotations, 4)
        elif prompt_type == PromptType.MASK:
            tensor_shape = (n, c, 256, 256)
        elif prompt_type == PromptType.POINT:
            max_annotations = utils.get_max_annotations(annotations)
            tensor_shape = (n, c, max_annotations, 2)

        tensor = torch.zeros(tensor_shape)
        flag = (
            torch.zeros(tensor_shape[:-1]).type(torch.uint8)
            if prompt_type != PromptType.MASK
            else torch.zeros(tensor_shape[:2]).type(torch.uint8)
        )

        if prompt_type == PromptType.MASK:
            for i, img_id in enumerate(annotations):
                for j, cat_id in enumerate(annotations[img_id]):
                    mask = self.prompts_processor.apply_masks(
                        annotations[img_id][cat_id]
                    )
                    tensor_mask = torch.tensor(mask)
                    tensor[i, j, :] = tensor_mask
                    flag[i, j] = 1 if torch.sum(tensor_mask) > 0 else 0
        else:
            for i, img_id in enumerate(annotations):
                img_original_size = (
                    self.images[img_id]["height"],
                    self.images[img_id]["width"],
                )
                for j, cat_id in enumerate(annotations[img_id]):
                    if annotations[img_id][cat_id].size == 0:
                        continue
                    m = annotations[img_id][cat_id].shape[0]
                    if prompt_type == PromptType.BBOX:
                        boxes_ann = self.prompts_processor.apply_boxes(
                            annotations[img_id][cat_id], img_original_size
                        )
                        tensor[i, j, :m, :] = torch.tensor(boxes_ann)
                    elif prompt_type == PromptType.POINT:
                        points_ann = self.prompts_processor.apply_coords(
                            annotations[img_id][cat_id], img_original_size
                        )
                        tensor[i, j, :m, :] = torch.tensor(points_ann)
                    flag[i, j, :m] = 1

        return tensor, flag


class LabelAnyThingOnlyImageDataset(Dataset):
    def __init__(self, directory=None, preprocess=None):
        super().__init__()
        self.directory = directory
        self.files = os.listdir(directory)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.directory, self.files[item]))
        image_id, _ = os.path.splitext(self.files[item])
        return self.preprocess(img), image_id  # load image


# main for testing the class
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor

    preprocess = Compose(
        [
            CustomResize(1024),
            PILToTensor(),
            CustomNormalize(),
        ]
    )
    dataset = CocoLVISDataset(
        instances_path="lvis_v1_train.json",
        max_num_examples=10,
        preprocess=preprocess,
    )

    """x = dataset[1]
    print([f'{k}: {v.size()}' for k, v in x.items() if isinstance(v, torch.Tensor)])
    exit()"""

    dataloader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn
    )
    data_dict, gt = next(iter(dataloader))

    print(
        [
            f"{k}: {v.size() if isinstance(v, torch.Tensor) else v}"
            for k, v in data_dict.items()
        ]
    )
    print(f"gt: {gt.size()}")
