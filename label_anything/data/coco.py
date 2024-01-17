import itertools
import os
import random
import warnings
from enum import IntEnum, Enum
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from safetensors.torch import load_file
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
    """
    Dataset for the COCO dataset.

    Args:
        name (str): Name of the dataset (coco or lvis).
        instances_path (str): Path to the COCO instances file.
        img_dir (str): Path to the directory containing the images.
        max_num_examples (int): Maximum number of examples for each image.
        preprocess (torchvision.transforms.Compose): Preprocess step.
        seed (int): Seed for reproducibility.
        emb_dir (str): Path to the directory containing the embeddings.
        n_folds (int): Number of folds for the FSS benchmark.
        val_fold (int): Validation fold for the FSS benchmark.
        load_embeddings (bool): Whether to load embeddings or images.
        split (str): Split for the FSS benchmark.
        do_subsample (bool): Whether to subsample the categories (True for train).
        add_box_noise (bool): Whether to add noise to the bounding boxes (True for train).
    """

    def __init__(
        self,
        name,  # dataset name (coco or lvis)
        instances_path,  # Path
        img_dir=None,  # directory (only if images have to be loaded from disk)
        max_num_examples=10,  # number of max examples to be given for the target image
        max_points_annotations=50,  # Max number of annotations for a class to be sampled as points or bboxes, if limit is reached it will be sampled as mask
        preprocess=ToTensor(),  # preprocess step
        seed=42,  # for reproducibility
        emb_dir=None,
        n_folds=-1,  # for fss benchmark (coco20i)
        val_fold=-1,  # for fss benchmark (coco20i)
        load_embeddings=False,
        load_gts=False,  # gts are in emb_dir files
        split="train",  # for fss benchmark (coco20i)
        do_subsample=True,
        add_box_noise=True,
    ):
        super().__init__()
        print(f"Loading dataset annotations from {instances_path}...")

        self.name = name
        instances = utils.load_instances(instances_path)

        self.emb_dir = emb_dir
        self.load_embeddings = load_embeddings
        self.load_gts = load_gts
        self.max_points_annotations = max_points_annotations
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
        if self.val_fold > -1:
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
        self.__set_all_seeds()

    def __set_all_seeds(self):
        """Enable reproducibility."""
        random.seed(self.seed)
        np.random.seed(int(self.seed))
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

    def _load_safe(self, img_data):
        f = load_file(f"{self.emb_dir}/{str(img_data['id']).zfill(12)}.safetensors")
        embedding, gt = None, None
        if self.load_embeddings:
            embedding = f["embedding"]
        if self.load_gts:
            gt = f[f"{self.name}_gt"]
        return embedding, gt

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

    def _extract_examples(self, img_data: dict, num_examples: int) -> (list, list):
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
            self.example_generator.sample_classes_from_query(img_cats, uniform_sampling)
            if self.do_subsample
            else img_cats
        )
        return self.example_generator.generate_examples(
            query_image_id=img_data["id"],
            image_classes=img_cats,
            sampled_classes=torch.tensor(sampled_classes),
            num_examples=num_examples,
        )

    def _get_annotations(
        self, image_ids: list, cat_ids: list
    ) -> (list, list, list, list, list):
        bboxes = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        masks = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        points = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]

        # get prompts from annotations
        classes = [list() for _ in image_ids]
        img_sizes = [
            (self.images[img_id]["height"], self.images[img_id]["width"])
            for img_id in image_ids
        ]

        for i, (img_id, img_size) in enumerate(zip(image_ids, img_sizes)):
            for cat_id in cat_ids:
                # for each pair (image img_id and category cat_id)
                if cat_id not in self.img2cat_annotations[img_id]:
                    # the chosen category is not in the iamge
                    continue

                classes[i].append(cat_id)
                number_of_annotations = len(self.img2cat_annotations[img_id][cat_id])
                if number_of_annotations > self.max_points_annotations:
                    # if there are too many annotations, sample a mask
                    prompt_types = [PromptType.MASK] * number_of_annotations
                else:
                    prompt_types = random.choices(
                        list(PromptType), k=number_of_annotations
                    )
                for ann, prompt_type in zip(self.img2cat_annotations[img_id][cat_id], prompt_types):
                    if prompt_type == PromptType.BBOX:
                        # take the bbox
                        bboxes[i][cat_id].append(
                            self.prompts_processor.convert_bbox(
                                ann["bbox"],
                                *img_size,
                                noise=self.add_box_noise,
                            ),
                        )
                    elif prompt_type == PromptType.MASK:
                        # take the mask
                        masks[i][cat_id].append(
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
                        points[i][cat_id].append(
                            self.prompts_processor.sample_point(mask)
                        )

        # convert the lists of prompts to arrays
        for i in range(len(image_ids)):
            for cat_id in cat_ids:
                bboxes[i][cat_id] = np.array((bboxes[i][cat_id]))
                masks[i][cat_id] = np.array((masks[i][cat_id]))
                points[i][cat_id] = np.array((points[i][cat_id]))
        return bboxes, masks, points, classes, img_sizes

    def _load_and_preprocess_image(self, image_data):
        image = self._load_image(image_data)
        return image if not self.preprocess else self.preprocess(image)

    def _get_images_or_embeddings(self, image_ids):
        if self.load_embeddings:
            embeddings_gts = [
                self._load_safe(image_data)
                for image_data in [self.images[image_id] for image_id in image_ids]
            ]
            embeddings, gts = zip(*embeddings_gts)
            if not self.load_gts:
                gts = None
            return torch.stack(embeddings), "embeddings", gts
        else:
            images = [
                self._load_and_preprocess_image(image_data)
                for image_data in [self.images[image_id] for image_id in image_ids]
            ]
            gts = None
            if self.load_gts:
                gts = [
                    self._load_safe(image_data)[1]
                    for image_data in [self.images[image_id] for image_id in image_ids]
                ]
            return torch.stack(images), "images", gts

    def __getitem__(self, idx_num_examples: tuple[int, int]) -> dict:
        idx, num_examples = idx_num_examples
        if self.split == "train":
            base_image_data = self.images[self.image_ids[idx]]
            image_ids, aux_cat_ids = self._extract_examples(
                base_image_data, num_examples
            )
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
        images, image_key, ground_truths = self._get_images_or_embeddings(image_ids)

        # create the prompt dicts
        bboxes, masks, points, classes, img_sizes = self._get_annotations(
            image_ids, cat_ids
        )

        # obtain padded tensors
        bboxes, flag_bboxes = self.annotations_to_tensor(
            bboxes, img_sizes, PromptType.BBOX
        )
        masks, flag_masks = self.annotations_to_tensor(
            masks, img_sizes, PromptType.MASK
        )
        points, flag_points = self.annotations_to_tensor(
            points, img_sizes, PromptType.POINT
        )

        # obtain ground truths
        if ground_truths is None:
            ground_truths = self.get_ground_truths(image_ids, cat_ids)

        # stack ground truths
        dims = torch.tensor(list(map(lambda x: x.size(), ground_truths)))
        max_dims = torch.max(dims, 0).values.tolist()
        ground_truths = torch.stack(
            [utils.collate_gts(x, max_dims) for x in ground_truths]
        )

        if self.load_gts:
            # convert the ground truths to the right format
            ground_truths_copy = ground_truths.clone()
            # set ground_truths to all 0s
            ground_truths = torch.zeros_like(ground_truths)
            for i, cat_id in enumerate(cat_ids):
                if cat_id == -1:
                    continue
                ground_truths[ground_truths_copy == cat_id] = i

        data_dict = {
            image_key: images,
            "prompt_masks": masks,
            "flag_masks": flag_masks,
            "prompt_points": points,
            "flag_points": flag_points,
            "prompt_bboxes": bboxes,
            "flag_bboxes": flag_bboxes,
            "dims": dims,
            "classes": classes,
            "image_ids": image_ids,
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
        ground_truths = [dict() for _ in image_ids]
        # generate masks
        for i, image_id in enumerate(image_ids):
            img_size = (self.images[image_id]["height"], self.images[image_id]["width"])
            for cat_id in cat_ids:
                ground_truths[i][cat_id] = np.zeros(img_size, dtype=np.int64)
                # zero mask for no segmentation
                if cat_id not in self.img2cat_annotations[image_id]:
                    continue
                for ann in self.img2cat_annotations[image_id][cat_id]:
                    ground_truths[i][cat_id] = np.logical_or(
                        ground_truths[i][cat_id],
                        self.prompts_processor.convert_mask(
                            ann["segmentation"], *img_size
                        ),
                    )
            # make the ground truth tensor for image img_id
            ground_truth = torch.from_numpy(
                np.array(
                    [ground_truths[i][cat_id].astype(np.int64) for cat_id in cat_ids]
                )
            )
            ground_truths[i] = torch.argmax(ground_truth, 0)

        return ground_truths

    def __len__(self):
        return len(self.images) if self.split == "train" else 1000

    def annotations_to_tensor(
        self, annotations: list, img_sizes: list, prompt_type: PromptType
    ) -> torch.Tensor:
        """Transform a dict of annotations of prompt_type to a padded tensor.

        Args:
            annotations (dict): annotations (dict of dicts with np.ndarray as values)
            prompt_type (PromptType): prompt type

        Returns:
            torch.Tensor: padded tensor
        """
        n = len(annotations)
        c = len(annotations[0])

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
            for i, annotation in enumerate(annotations):
                for j, cat_id in enumerate(annotation):
                    mask = self.prompts_processor.apply_masks(annotation[cat_id])
                    tensor_mask = torch.tensor(mask)
                    tensor[i, j, :] = tensor_mask
                    flag[i, j] = 1 if torch.sum(tensor_mask) > 0 else 0
        else:
            for i, (annotation, img_original_size) in enumerate(
                zip(annotations, img_sizes)
            ):
                for j, cat_id in enumerate(annotation):
                    if annotation[cat_id].size == 0:
                        continue
                    m = annotation[cat_id].shape[0]
                    if prompt_type == PromptType.BBOX:
                        boxes_ann = self.prompts_processor.apply_boxes(
                            annotation[cat_id], img_original_size
                        )
                        tensor[i, j, :m, :] = torch.tensor(boxes_ann)
                    elif prompt_type == PromptType.POINT:
                        points_ann = self.prompts_processor.apply_coords(
                            annotation[cat_id], img_original_size
                        )
                        tensor[i, j, :m, :] = torch.tensor(points_ann)
                    flag[i, j, :m] = 1

        return tensor, flag


class CocoLVISTestDataset(CocoLVISDataset):
    def __init__(
            self,
            name,
            instances_path,  # Path
            img_dir=None,  # directory (only if images have to be loaded from disk)
            max_num_examples=10,  # number of max examples to be given for the target image
            preprocess=ToTensor(),  # preprocess step
            emb_dir=None,
            seed=42,  # for reproducibility
            load_embeddings=False,  # max number of coords for each example for each class
            load_gts=False,
            add_box_noise=False,
    ):
        super(CocoLVISTestDataset, self).__init__(name=name,
                                                  instances_path=instances_path,
                                                  img_dir=img_dir,
                                                  max_num_examples=max_num_examples,
                                                  preprocess=preprocess,
                                                  seed=seed,
                                                  add_box_noise=add_box_noise,
                                                  emb_dir=emb_dir,
                                                  load_embeddings=load_embeddings,
                                                  load_gts=load_gts,)

    def _extract_examples(
            self,
            cat2img: dict,
            img2cat: dict
    ) -> list[int]:
        prompt_images = set()
        for cat_id in self.categories.keys():
            if cat_id not in cat2img:
                continue
            cat_images = cat2img[cat_id]
            _, img = max(map(lambda x: (len(img2cat[x]), x), cat_images))
            prompt_images.add(img)
        return prompt_images

    def _get_images_or_embeddings(self, image_ids):
        if self.load_embeddings:
            embeddings_gts = [
                self._load_safe(data)
                for data in image_ids
            ]
            embeddings, gts = zip(*embeddings_gts)
            if not self.load_gts:
                gts = None
            return torch.stack(embeddings), "embeddings", gts
        images = [
            self._load_and_preprocess_image(image_data)
            for image_data in image_ids
        ]
        gts = None
        if self.load_gts:
            gts = [
                self._load_safe(image_data)[1]
                for image_data in image_ids
            ]
        return torch.stack(images), "images", gts

    def extract_prompts(
            self,
            cat2img: dict,
            img2cat: dict,
            images: dict,
            img2cat_annotations: dict,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        image_ids = self._extract_examples(cat2img, img2cat)
        prompt_images = [images[x] for x in image_ids]

        prompt_images, prompt_images_key, _ = self._get_images_or_embeddings(prompt_images)

        cat_ids = list(self.categories.keys())
        random.shuffle(cat_ids)
        bboxes, masks, points, _, image_sizes = self._get_annotations(image_ids, cat_ids,
                                                                      images, img2cat_annotations)

        bboxes, flag_bboxes = self.annotations_to_tensor(bboxes, image_sizes, PromptType.BBOX)
        masks, flag_masks = self.annotations_to_tensor(masks, image_sizes, PromptType.MASK)
        points, flag_points = self.annotations_to_tensor(points, image_sizes, PromptType.POINT)
        return {
            prompt_images_key: prompt_images,
            "prompt_masks": masks,
            "flag_masks": flag_masks,
            "prompt_points": points,
            "flag_points": flag_points,
            "prompt_bboxes": bboxes,
            "flag_bboxes": flag_bboxes,
            "dims": torch.as_tensor(image_sizes),
        }

    def _get_annotations(
        self,
        image_ids: list,
        cat_ids: list,
        images: dict,
        img2cat_annotations: dict,
    ) -> (list, list, list, list, list):
        bboxes = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        masks = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        points = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]

        # get prompts from annotations
        classes = [list() for _ in image_ids]
        img_sizes = [
            (images[img_id]["height"], images[img_id]["width"])
            for img_id in image_ids
        ]

        for i, (img_id, img_size) in enumerate(zip(image_ids, img_sizes)):
            for cat_id in cat_ids:
                # for each pair (image img_id and category cat_id)
                if cat_id not in img2cat_annotations[img_id]:
                    # the chosen category is not in the iamge
                    continue

                classes[i].append(cat_id)
                number_of_annotations = len(img2cat_annotations[img_id][cat_id])
                if number_of_annotations > self.max_points_annotations:
                    # if there are too many annotations, sample a mask
                    prompt_types = [PromptType.MASK] * number_of_annotations
                else:
                    prompt_types = random.choices(
                        list(PromptType), k=number_of_annotations
                    )
                for ann, prompt_type in zip(img2cat_annotations[img_id][cat_id], prompt_types):
                    if prompt_type == PromptType.BBOX:
                        # take the bbox
                        bboxes[i][cat_id].append(
                            self.prompts_processor.convert_bbox(
                                ann["bbox"],
                                *img_size,
                                noise=self.add_box_noise,
                            ),
                        )
                    elif prompt_type == PromptType.MASK:
                        # take the mask
                        masks[i][cat_id].append(
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
                        points[i][cat_id].append(
                            self.prompts_processor.sample_point(mask)
                        )

        # convert the lists of prompts to arrays
        for i in range(len(image_ids)):
            for cat_id in cat_ids:
                bboxes[i][cat_id] = np.array((bboxes[i][cat_id]))
                masks[i][cat_id] = np.array((masks[i][cat_id]))
                points[i][cat_id] = np.array((points[i][cat_id]))
        return bboxes, masks, points, classes, img_sizes

    def __getitem__(self, item):
        data, data_key, gt = self._get_images_or_embeddings([self.images[self.image_ids[item]]])
        gt = torch.stack(gt)
        dim = torch.as_tensor(gt.size())
        data_dict = {
            data_key: data,
            "dim": dim,
            "gt": gt,
        }
        return data_dict

    def collate_fn(
        self, batched_input: List[Dict[str, Any]]
    ) -> (Dict[str, Any], torch.Tensor):
        data_key = 'images' if 'images' in batched_input[0].keys() else 'embeddings'
        images = torch.stack([x[data_key] for x in batched_input])

        dims = torch.stack([x["dim"] for x in batched_input])

        max_dims = torch.max(dims, 0).values.tolist()
        gt = torch.stack([utils.collate_gts(x["gt"], max_dims) for x in batched_input])

        data_dict = {
            data_key: images,
            "dims": dims,
        }

        return data_dict, gt


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
    dataset = CocoLVISTestDataset(
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
