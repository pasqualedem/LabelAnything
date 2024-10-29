import json
import os
import random
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import numpy as np
import torch
from scipy.ndimage import label, binary_dilation
from label_anything.data.coco20i import Coco20iDataset
from safetensors.torch import load_file
import itertools
from torchvision.transforms import PILToTensor, ToTensor

import label_anything.data.utils as utils
from label_anything.data.utils import (
    AnnFileKeys,
    BatchKeys,
    BatchMetadataKeys,
    PromptType,
    flags_merge,
)
from label_anything.data.transforms import PromptsProcessor
from label_anything.data.test import LabelAnythingTestDataset
from label_anything.data.examples import build_example_generator, uniform_sampling
from label_anything.logger.text_logger import get_logger
import os

logger = get_logger(__name__)


class PascalDataset(Dataset):
    """Pascal VOC dataset."""

    def __init__(
        self,
        name: str,
        data_dir: str,  # data/pascal
        split: str,  # data/pascal/ImageSets/Segmentation/train.txt
        emb_dir: Optional[str] = None,  # data/pascal/vit_sam_embeddings
        n_ways: int = "max",
        preprocess=ToTensor(),
        image_size: int = 1024,
        load_embeddings: bool = None,
        load_gts: bool = False,
        do_subsample: bool = True,
        remove_small_annotations: bool = False,
        all_example_categories: bool = True,
        sample_function: str = "power_law",
        custom_preprocess: bool = True,
    ):
        super().__init__()
        print(f"Loading image filenames from {split}...")

        assert (
            not load_gts or emb_dir is not None
        ), "If load_gts is True, emb_dir must be provided."
        assert (
            not load_embeddings or emb_dir is not None
        ), "If load_embeddings is True, emb_dir must be provided."

        if load_embeddings is None:
            load_embeddings = emb_dir is not None
            logger.warning(
                f"load_embeddings is not specified. Assuming load_embeddings={load_embeddings}."
            )
        self.name = name
        self.split = split
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "JPEGImages")
        self.masks_dir = os.path.join(data_dir, "SegmentationClass")
        self.emb_dir = emb_dir
        self.n_ways = n_ways
        self.image_size = image_size
        self.load_embeddings = load_embeddings
        self.all_example_categories = all_example_categories
        self.load_gts = load_gts
        self.do_subsample = do_subsample
        self.remove_small_annotations = remove_small_annotations
        self.sample_function = sample_function

        self.masks_dir_list = set(os.listdir(self.masks_dir))
        self.aug_masks_dir_list = set(os.listdir(self.masks_dir + "Aug"))

        # read the image names
        self.image_names = []
        if split == "train":
            filenames_path1 = os.path.join(
                os.path.join(data_dir, "ImageSets/Segmentation/train.txt")
            )
            filenames_path2 = os.path.join(
                os.path.join(data_dir, "ImageSets/Segmentation/trainaug.txt")
            )
            filenames_paths = [filenames_path1, filenames_path2]
            for filenames_path in filenames_paths:
                with open(filenames_path) as f:
                    for line in f:
                        image_name = line.rstrip()
                        self.image_names.append(image_name)
            # remove duplicates without changing the order
            self.image_names = list(dict.fromkeys(self.image_names))
        elif split == "val":
            filenames_path = os.path.join(data_dir, "ImageSets/Segmentation/val.txt")
            with open(filenames_path) as f:
                for line in f:
                    image_name = line.rstrip()
                    self.image_names.append(image_name)

        # read the categories
        self.categories = {
            1: {"name": "aeroplane"},
            2: {"name": "bicycle"},
            3: {"name": "bird"},
            4: {"name": "boat"},
            5: {"name": "bottle"},
            6: {"name": "bus"},
            7: {"name": "car"},
            8: {"name": "cat"},
            9: {"name": "chair"},
            10: {"name": "cow"},
            11: {"name": "diningtable"},
            12: {"name": "dog"},
            13: {"name": "horse"},
            14: {"name": "motorbike"},
            15: {"name": "person"},
            16: {"name": "pottedplant"},
            17: {"name": "sheep"},
            18: {"name": "sofa"},
            19: {"name": "train"},
            20: {"name": "tvmonitor"},
        }

        self.img2cat, self.cat2img = self._load_annotation_dicts()

        # example generator/selector
        self.example_generator = build_example_generator(
            n_ways=self.n_ways,
            n_shots=None,
            images_to_categories=self.img2cat,
            categories_to_imgs=self.cat2img,
            sample_function=self.sample_function,
        )

        # processing
        self.preprocess = preprocess
        self.prompts_processor = PromptsProcessor(
            long_side_length=self.image_size,
            masks_side_length=256,
            custom_preprocess=custom_preprocess,
        )

    def __get_seg(self, input_str: str, with_random_choice: bool = True):
        image_path, seg_path = input_str.split()
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        seg = None
        seg_aug = None
        
        if os.path.basename(seg_path) in self.masks_dir_list:
            seg_filename = os.path.join(self.masks_dir, os.path.basename(seg_path))
            seg = Image.open(seg_filename)
            seg = np.array(seg, dtype=np.int64)
        
        if self.split == "val":
            return seg
        
        aug_seg_path = os.path.join(self.masks_dir + "Aug", os.path.basename(seg_path))
        if os.path.basename(seg_path) in self.aug_masks_dir_list:
            seg_aug = Image.open(aug_seg_path)
            seg_aug = np.array(seg_aug, dtype=np.int64)

        if self.remove_small_annotations:
            if seg is not None:
                for cat_id in np.unique(seg):
                    mask = seg == cat_id
                    if np.sum(mask) < 2 * 32 * 32:
                        seg[mask] = 0
            if seg_aug is not None:
                for cat_id in np.unique(seg_aug):
                    mask = seg_aug == cat_id
                    if np.sum(mask) < 2 * 32 * 32:
                        seg_aug[mask] = 0
        
        return seg, seg_aug

    # def __get_seg(self, image_name: str, with_random_choice: bool = True):
    #     seg = None
    #     seg_aug = None
    #     if image_name + ".png" in self.masks_dir_list:
    #         seg_filename = os.path.join(self.masks_dir, image_name + ".png")
    #         seg = Image.open(seg_filename)
    #         seg = np.array(seg, dtype=np.int64)
    #     if self.split == "val":
    #         return seg
    #     if image_name + ".png" in self.aug_masks_dir_list:
    #         seg_filename = os.path.join(self.masks_dir + "Aug", image_name + ".png")
    #         seg_aug = Image.open(seg_filename)
    #         seg_aug = np.array(seg_aug, dtype=np.int64)

    #     if self.remove_small_annotations:
    #         if seg is not None:
    #             for cat_id in np.unique(seg):
    #                 # count number of pixels of cat_id
    #                 mask = seg == cat_id
    #                 if np.sum(mask) < 2 * 32 * 32:
    #                     seg[mask] = 0
    #         if seg_aug is not None:
    #             for cat_id in np.unique(seg_aug):
    #                 # count number of pixels of cat_id
    #                 mask = seg_aug == cat_id
    #                 if np.sum(mask) < 2 * 32 * 32:
    #                     seg_aug[mask] = 0

    #     if seg is None and seg_aug is not None:
    #         return seg_aug
    #     elif seg is not None and seg_aug is None:
    #         return seg
    #     else:
    #         # randomly choose between seg and seg_aug
    #         if with_random_choice:
    #             return seg if random.random() < 0.5 else seg_aug
    #         else:
    #             return seg

    def _load_annotation_dicts(self):
        img2cat = {}
        cat2img = {}

        for image_name in self.image_names:
            seg = self.__get_seg(image_name, with_random_choice=False)
            categories = np.unique(seg[(seg != 0) & (seg != 255)]).tolist()
            categories = [cat for cat in categories if cat in self.categories]
            img2cat[image_name] = categories
            for cat in categories:
                if cat not in cat2img:
                    cat2img[cat] = set()
                cat2img[cat].add(image_name)

        return img2cat, cat2img

    def __len__(self):
        return len(self.image_names)
    
    def load_and_preprocess_images(self, image_names: list[str]) -> torch.Tensor:
        images = [
            Image.open(f"{self.img_dir}/{image_name}.jpg")
            for image_name in image_names
        ]
        if self.preprocess is not None:
            images = [self.preprocess(image) for image in images]
        return images

    def _extract_examples(
        self, image_name: str, num_examples: int, num_classes: int
    ) -> (list[int], list[int]):
        """Chooses examples (and categories) for the query image.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.
            num_examples (int): The number of examples to be chosen.

        Returns:
            (list, list): Returns two lists:
                1. examples: A list of image ids of the examples.
                2. cats: A list of sets of category ids of the examples.
        """
        img_cats = torch.tensor(list(self.img2cat[image_name]))
        sampled_classes = (
            self.example_generator.sample_classes_from_query(img_cats, uniform_sampling)
            if self.do_subsample
            else img_cats
        )
        return self.example_generator.generate_examples(
            query_image_id=image_name,
            image_classes=img_cats,
            sampled_classes=torch.tensor(sampled_classes),
            num_examples=num_examples,
            num_classes=num_classes,
        )

    def _load_safe(self, img_name: str) -> (torch.Tensor, Optional[torch.Tensor]):
        """Open a safetensors file and load the embedding and the ground truth.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            (torch.Tensor, Optional[torch.Tensor]): Returns a tuple containing the embedding and the ground truth.
        """
        assert self.emb_dir is not None, "emb_dir must be provided."
        gt = None

        f = load_file(
            f"{self.emb_dir}/{img_name}.safetensors"
        )
        embedding = f["embedding"]
        if self.load_gts:
            gt = f[f"{self.name}_gt"]
        return embedding, gt

    def _get_images_or_embeddings(
        self, image_names: list[str]
    ) -> (torch.Tensor, str, Optional[torch.Tensor]):
        """Load, stack and preprocess the images or the embeddings.

        Args:
            image_ids (list[int]): A list of image ids.

        Returns:
            (torch.Tensor, str, Optional[torch.Tensor]): Returns a tuple containing the images or the embeddings, the key of the returned tensor and the ground truths.
        """
        if self.load_embeddings:
            embeddings_gts = [
                self._load_safe(image_name)
                for image_name in image_names
            ]
            embeddings, gts = zip(*embeddings_gts)
            if not self.load_gts:
                gts = None
            return torch.stack(embeddings), BatchKeys.EMBEDDINGS, gts
        else:
            images = [
                Image.open(f"{self.img_dir}/{image_name}.jpg")
                for image_name in image_names
            ]
            if self.preprocess is not None:
                images = [self.preprocess(image) for image in images]
            gts = None
            return torch.stack(images), BatchKeys.IMAGES, gts

    def _get_prompts(
        self, image_names: list, cat_ids: list, with_random_choice: bool = True
    ) -> (list, list, list, list, list):
        """Get the annotations for the chosen examples.

        Args:
            image_names (list): A list of image ids of the examples.
            cat_ids (list): A list of sets of category ids of the examples.

        Returns:
            (list, list, list, list, list): Returns five lists:
                2. masks: A list of dictionaries mapping category ids to masks.
        """
        masks = [{cat_id: [] for cat_id in cat_ids} for _ in image_names]

        classes = [[] for _ in range(len(image_names))]
        # it wont work if we have more than one example per image
        segs = [
            self.__get_seg(image_name, with_random_choice=with_random_choice)
            for image_name in image_names
        ]
        img_sizes = [image.shape[-2:] for image in segs]

        # process annotations
        for i, (img_name, img_size) in enumerate(zip(image_names, img_sizes)):
            for cat_id in cat_ids:
                # for each pair (image img_id and category cat_id)
                if cat_id not in self.img2cat[img_name]:
                    continue
                classes[i].append(cat_id)

                # get the annotation
                seg = segs[i]

                # create the binary mask where seg == cat_id
                mask = np.zeros_like(seg)
                mask[seg == cat_id] = 1

                masks[i][cat_id].append(mask)

        # convert the lists of prompts to arrays
        for i in range(len(image_names)):
            for cat_id in cat_ids:
                masks[i][cat_id] = np.array((masks[i][cat_id]))
        return masks, classes, img_sizes

    def compute_ground_truths(
        self, image_names: list[str], img_sizes, cat_ids: list[int], with_random_choice: bool = True
    ) -> list[torch.Tensor]:
        """Compute the ground truths for the given image ids and category ids.

        Args:
            image_ids (list[int]): Image ids.
            cat_ids (list[int]): Category ids.

        Returns:
            list[torch.Tensor]: A list of tensors containing the ground truths (per image).
        """
        ground_truths = []

        # generate masks
        for i, image_name in enumerate(image_names):
            img_size = img_sizes[i]
            ground_truths.append(np.zeros(img_size, dtype=np.int64))
            seg = self.__get_seg(image_name, with_random_choice=with_random_choice)

            for cat_id in cat_ids:
                if cat_id not in self.img2cat[image_name]:
                    continue
                mask = seg == cat_id
                ground_truths[-1][mask] = cat_ids.index(cat_id)

        return [torch.tensor(x) for x in ground_truths]

    def __getitem__(self, idx_metadata: tuple[int, int]) -> dict:
        """Get an item from the dataset.

        Args:
            idx_metadata (tuple[int, dict]): A tuple containing the index of the image and the batch level metadata e.g. number of examples to be chosen and type of prompts.

        Returns:
            dict: A dictionary containing the data.
        """
        idx, batch_metadata = idx_metadata

        num_examples = batch_metadata[BatchMetadataKeys.NUM_EXAMPLES]
        possible_prompt_types = batch_metadata[BatchMetadataKeys.PROMPT_TYPES]
        if batch_metadata[BatchMetadataKeys.PROMPT_CHOICE_LEVEL] == "episode":
            possible_prompt_types = random.choice(possible_prompt_types)
        num_classes = batch_metadata.get(BatchMetadataKeys.NUM_CLASSES, None)

        image_name = self.image_names[idx]
        image_names, aux_cat_ids = self._extract_examples(
            image_name, num_examples, num_classes
        )

        if self.all_example_categories:
            aux_cat_ids = [aux_cat_ids[0]] + [
                set(self.img2cat[img]) for img in image_names[1:]
            ]  # check if self.images must be called before

        cat_ids = sorted(list(set(itertools.chain(*aux_cat_ids))))
        cat_ids.insert(0, -1)

        # load, stack and preprocess the images
        images, image_key, ground_truths = self._get_images_or_embeddings(image_names)

        masks, classes, img_sizes = self._get_prompts(image_names, cat_ids)

        masks, flag_masks = utils.annotations_to_tensor(
            self.prompts_processor, masks, img_sizes, PromptType.MASK
        )

        if ground_truths is None:
            ground_truths = self.compute_ground_truths(image_names, img_sizes, cat_ids)

        # stack ground truths
        dims = torch.tensor(img_sizes)
        max_dims = torch.max(dims, 0).values.tolist()
        ground_truths = torch.stack(
            [utils.collate_gts(x, max_dims) for x in ground_truths]
        )

        if self.load_gts:
            # convert the ground truths to the right format
            # by assigning 0 to n-1 to the classes
            ground_truths_copy = ground_truths.clone()
            # set ground_truths to all 0s
            ground_truths = torch.zeros_like(ground_truths)
            for i, cat_id in enumerate(cat_ids):
                if cat_id == -1:
                    continue
                ground_truths[ground_truths_copy == cat_id] = i

        # make zeroes tensors for boxes, points and flags
        prompt_bboxes = torch.zeros(
            (len(image_names), len(cat_ids), 1, 4), dtype=torch.float32
        )
        flag_bboxes = torch.zeros(
            (len(image_names), len(cat_ids), 1), dtype=torch.uint8
        )
        prompt_points = torch.zeros(
            (len(image_names), len(cat_ids), 1, 2), dtype=torch.float32
        )
        flag_points = torch.zeros(
            (len(image_names), len(cat_ids), 1), dtype=torch.uint8
        )
        
        flag_examples = flags_merge(flag_masks, flag_points, flag_bboxes)

        data_dict = {
            image_key: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.PROMPT_BBOXES: prompt_bboxes,
            BatchKeys.FLAG_BBOXES: flag_bboxes,
            BatchKeys.PROMPT_POINTS: prompt_points,
            BatchKeys.FLAG_POINTS: flag_points,
            BatchKeys.DIMS: dims,
            BatchKeys.CLASSES: classes,
            BatchKeys.IMAGE_IDS: image_names,
            BatchKeys.GROUND_TRUTHS: ground_truths,
        }
        return data_dict
