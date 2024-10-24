import itertools
import os
import random
import warnings
from io import BytesIO
from typing import Any, Optional

import numpy as np
import requests
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, ToTensor
from tap.logger.text_logger import get_logger

import tap.data.utils as utils
from tap.data.transforms import (
    CustomNormalize,
    CustomResize,
    PromptsProcessor,
)
from tap.data.utils import (
    AnnFileKeys,
    BatchKeys,
    BatchMetadataKeys,
    PromptType,
    flags_merge,
)
from tap.data.test import LabelAnythingTestDataset

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


class CocoLVISDataset(Dataset):
    """Dataset class for COCO and LVIS datasets."""

    def __init__(
        self,
        name: str,
        instances_path: str,
        img_dir: Optional[str] = None,
        emb_dir: Optional[str] = None,
        max_points_per_annotation: int = 10,
        max_points_annotations: int = 50,
        n_ways: int = "max",
        preprocess=ToTensor(),
        image_size: int = 1024,
        load_embeddings: bool = None,
        load_gts: bool = False,
        do_subsample: bool = True,
        add_box_noise: bool = True,
        remove_small_annotations: bool = False,
        all_example_categories: bool = True,
        sample_function: str = "power_law",
        custom_preprocess: bool = True,
    ):
        """Initialize the dataset.

        Args:
            name (str): A name for the dataset (e.g. "coco", "lvis").
            instances_path (str): Path to the instances json file.
            img_dir (Optional[str], optional): Path to the directory containing the images. Defaults to None.
            emb_dir (Optional[str], optional): Path to the directory containing the embeddings. Defaults to None.
            max_points_per_annotation (int, optional): Maximum number of points per annotation. Defaults to 10.
            max_points_annotations (int, optional): Maximum number of sparse prompts. Defaults to 50.
            n_ways (int, optional): Number of classes to sample. Defaults to "max".
            preprocess (_type_, optional): A preprocessing step to apply to the images. Defaults to ToTensor().
            load_embeddings (bool, optional): Specify if embeddings are precomputed. Defaults to True.
            load_gts (bool, optional): Specify if ground truth masks are precomputed. Defaults to False.
            do_subsample (bool, optional): Specify if classes should be randomly subsampled. Defaults to True.
            add_box_noise (bool, optional): Add noise to the boxes (useful for training). Defaults to True.
            prompt_types (list[PromptType], optional): List of prompt types to be used. Defaults to [PromptType.BBOX, PromptType.MASK, PromptType.POINT].
            all_example_categories (bool, optional): Specify if all exaple categories are taken into account.
            sample_function (str, optional): Specify strategy to sample support images.
            custom_preprocess (bool, optional): Specify if custom preprocessing is used. Defaults to True.
        """
        super().__init__()
        print(f"Loading dataset annotations from {instances_path}...")

        assert (
            img_dir is not None or emb_dir is not None
        ), "Either img_dir or emb_dir must be provided."
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
        self.instances_path = instances_path

        self.img_dir = img_dir
        self.emb_dir = emb_dir
        self.load_embeddings = load_embeddings
        self.load_gts = load_gts
        self.max_points_per_annotation = max_points_per_annotation
        self.max_points_annotations = max_points_annotations
        self.do_subsample = do_subsample
        self.add_box_noise = add_box_noise
        self.n_ways = n_ways
        self.image_size = image_size
        self.remove_small_annotations = remove_small_annotations
        self.all_example_categories = all_example_categories
        self.sample_function = sample_function

        # load instances
        instances = utils.load_instances(self.instances_path)
        self.annotations = {
            x[AnnFileKeys.ID]: x for x in instances[AnnFileKeys.ANNOTATIONS]
        }
        self.categories = {
            x[AnnFileKeys.ID]: x for x in instances[AnnFileKeys.CATEGORIES]
        }

        # useful dicts
        (
            self.img_annotations,
            self.img2cat,
            self.img2cat_annotations,
            self.cat2img,
            self.cat2img_annotations,
        ) = self._load_annotation_dicts()

        # load image ids and info
        img2cat_keys = set(self.img2cat.keys())
        self.images = {
            x[AnnFileKeys.ID]: x
            for x in instances[AnnFileKeys.IMAGES]
            if x[AnnFileKeys.ID] in img2cat_keys
        }
        self.image_ids = list(self.images.keys())

        # processing
        self.preprocess = preprocess
        self.prompts_processor = PromptsProcessor(
            long_side_length=self.image_size,
            masks_side_length=256,
            custom_preprocess=custom_preprocess,
        )

    def _load_annotation_dicts(self) -> tuple[dict, dict, dict, dict, dict]:
        """Load useful annotation dicts.

        Returns:
            (dict, dict, dict, dict, dict): Returns four dictionaries:
                0. img_annotations: A dictionary mapping image ids to lists of annotations.
                1. img2cat: A dictionary mapping image ids to sets of category ids.
                2. img2cat_annotations: A dictionary mapping image ids to dictionaries mapping category ids to annotations.
                3. cat2img: A dictionary mapping category ids to sets of image ids.
                4. cat2img_annotations: A dictionary mapping category ids to dictionaries mapping image ids to annotations.
        """
        img_annotations = {}
        img2cat_annotations = {}
        cat2img_annotations = {}

        img2cat = {}
        cat2img = {}

        category_ids = set(self.categories.keys())

        for ann in self.annotations.values():
            if self._remove_small_annotations(ann):
                continue

            if AnnFileKeys.ISCROWD in ann and ann[AnnFileKeys.ISCROWD] == 1:
                continue

            if ann[AnnFileKeys.CATEGORY_ID] not in category_ids:
                continue

            if ann[AnnFileKeys.IMAGE_ID] not in img_annotations:
                img_annotations[ann[AnnFileKeys.IMAGE_ID]] = []
            img_annotations[ann[AnnFileKeys.IMAGE_ID]].append(ann)

            if ann[AnnFileKeys.IMAGE_ID] not in img2cat_annotations:
                img2cat_annotations[ann[AnnFileKeys.IMAGE_ID]] = {}
                img2cat[ann[AnnFileKeys.IMAGE_ID]] = set()

            if (
                ann[AnnFileKeys.CATEGORY_ID]
                not in img2cat_annotations[ann[AnnFileKeys.IMAGE_ID]]
            ):
                img2cat_annotations[ann[AnnFileKeys.IMAGE_ID]][
                    ann[AnnFileKeys.CATEGORY_ID]
                ] = []
                img2cat[ann[AnnFileKeys.IMAGE_ID]].add(ann[AnnFileKeys.CATEGORY_ID])

            img2cat_annotations[ann[AnnFileKeys.IMAGE_ID]][
                ann[AnnFileKeys.CATEGORY_ID]
            ].append(ann)

            if ann[AnnFileKeys.CATEGORY_ID] not in cat2img_annotations:
                cat2img_annotations[ann[AnnFileKeys.CATEGORY_ID]] = {}
                cat2img[ann[AnnFileKeys.CATEGORY_ID]] = set()

            if ann["image_id"] not in cat2img_annotations[ann[AnnFileKeys.CATEGORY_ID]]:
                cat2img_annotations[ann[AnnFileKeys.CATEGORY_ID]][
                    ann[AnnFileKeys.IMAGE_ID]
                ] = []
                cat2img[ann[AnnFileKeys.CATEGORY_ID]].add(ann[AnnFileKeys.IMAGE_ID])

            cat2img_annotations[ann[AnnFileKeys.CATEGORY_ID]][
                ann[AnnFileKeys.IMAGE_ID]
            ].append(ann)
        return (
            img_annotations,
            img2cat,
            img2cat_annotations,
            cat2img,
            cat2img_annotations,
        )

    def _load_safe(self, img_data: dict) -> (torch.Tensor, Optional[torch.Tensor]): # type: ignore
        """Open a safetensors file and load the embedding and the ground truth.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            (torch.Tensor, Optional[torch.Tensor]): Returns a tuple containing the embedding and the ground truth.
        """
        assert self.emb_dir is not None, "emb_dir must be provided."
        f = load_file(
            f"{self.emb_dir}/{str(img_data[AnnFileKeys.ID]).zfill(12)}.safetensors"
        )
        embedding = f["embedding"]
        gt = f[f"{self.name}_gt"] if self.load_gts else None
        return embedding, gt

    def _load_image(self, img_data: dict) -> Image:
        """Load an image from disk or from url.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            PIL.Image: The loaded image.
        """
        if self.img_dir is not None:
            return Image.open(f'{self.img_dir}/{img_data["file_name"]}').convert("RGB")
        return Image.open(BytesIO(requests.get(img_data["coco_url"]).content)).convert(
            "RGB"
        )

    def _load_and_preprocess_image(self, img_data: dict) -> torch.Tensor:
        """Load and preprocess an image.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            torch.Tensor: The preprocessed image.
        """
        image = self._load_image(img_data)
        return self.preprocess(image) if self.preprocess else image

    def load_and_preprocess_images(self, img_ids: list[int]) -> torch.Tensor:
        """Load and preprocess images.

        Args:
            img_ids (list[int]): A list of image ids.
        Returns:
            torch.Tensor: The preprocessed images.
        """
        return [
            self._load_and_preprocess_image(self.images[img_id]) for img_id in img_ids
        ]

    def _sample_num_points(self, image_id: int, ann: dict) -> int:
        """
        Calculate the number of points to sample for a given image and category proportionally to the area of the annotation.

        Args:
            image_id (int): The ID of the image.
            ann (dict): The annotation.

        Returns:
            int: The number of points to sample.
        """
        image_area = self.images[image_id]["height"] * self.images[image_id]["width"]
        annotation_area = ann["area"] / image_area
        poisson_mean = self.max_points_per_annotation * np.sqrt(
            annotation_area
        )  # poisson mean is proportional to the square root of the area
        return np.clip(
            np.random.poisson(poisson_mean) + 1, 1, self.max_points_per_annotation
        )

    def _remove_small_annotations(self, ann: dict) -> bool:
        """Remove annotation smaller than 2*32*32 pixels.

        Args:
            ann (dict): The annotation.

        Returns:
            bool: True if the annotation is too small, False otherwise.
        """
        return ann["area"] < 2 * 32 * 32 if self.remove_small_annotations else False

    def _get_prompts(
        self, image_ids: list, cat_ids: list, possible_prompt_types: list[PromptType]
    ) -> (list, list, list, list, list): # type: ignore
        """Get the annotations for the chosen examples.

        Args:
            image_ids (list): A list of image ids of the examples.
            cat_ids (list): A list of sets of category ids of the examples.
            possible_prompt_types (list[PromptType]): A list of possible prompt types to be sampled.

        Returns:
            (list, list, list, list, list): Returns five lists:
                1. bboxes: A list of dictionaries mapping category ids to bounding boxes.
                2. masks: A list of dictionaries mapping category ids to masks.
                3. points: A list of dictionaries mapping category ids to points.
                4. classes: A list of lists of category ids.
                5. img_sizes: A list of tuples containing the image sizes.
        """
        bboxes = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        masks = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        points = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]

        classes = [[] for _ in range(len(image_ids))]
        img_sizes = [
            (self.images[img_id]["height"], self.images[img_id]["width"])
            for img_id in image_ids
        ]

        # process annotations
        for i, (img_id, img_size) in enumerate(zip(image_ids, img_sizes)):
            for cat_id in cat_ids:
                # for each pair (image img_id and category cat_id)
                if cat_id not in self.img2cat_annotations[img_id]:
                    continue
                classes[i].append(cat_id)

                # get the prompt type for each annotation
                n_ann = len(self.img2cat_annotations[img_id][cat_id])
                if n_ann > self.max_points_annotations:
                    prompt_types = [PromptType.MASK] * n_ann
                else:
                    prompt_types = random.choices(possible_prompt_types, k=n_ann)
                for ann, prompt_type in zip(
                    self.img2cat_annotations[img_id][cat_id], prompt_types
                ):
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
                                ann[AnnFileKeys.SEGMENTATION],
                                *img_size,
                            )
                        )
                    elif prompt_type == PromptType.POINT:
                        # take the point
                        mask = self.prompts_processor.convert_mask(
                            ann[AnnFileKeys.SEGMENTATION],
                            *img_size,
                        )
                        num_points = self._sample_num_points(img_id, ann)
                        for _ in range(num_points):
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

    def _get_images_or_embeddings(
        self, image_ids: list[int]
    ) -> (torch.Tensor, str, Optional[torch.Tensor]): # type: ignore
        """Load, stack and preprocess the images or the embeddings.

        Args:
            image_ids (list[int]): A list of image ids.

        Returns:
            (torch.Tensor, str, Optional[torch.Tensor]): Returns a tuple containing the images or the embeddings, the key of the returned tensor and the ground truths.
        """
        if self.load_embeddings:
            embeddings_gts = [
                self._load_safe(image_data)
                for image_data in [self.images[image_id] for image_id in image_ids]
            ]
            embeddings, gts = zip(*embeddings_gts)
            if not self.load_gts:
                gts = None
            return torch.stack(embeddings), BatchKeys.EMBEDDINGS, gts
        else:
            images = [
                self._load_and_preprocess_image(image_data)
                for image_data in [self.images[image_id] for image_id in image_ids]
            ]
            gts = None
            return torch.stack(images), BatchKeys.IMAGES, gts

    def compute_ground_truths(
        self, image_ids: list[int], cat_ids: list[int]
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
        for i, image_id in enumerate(image_ids):
            img_size = (self.images[image_id]["height"], self.images[image_id]["width"])
            ground_truths.append(np.zeros(img_size, dtype=np.int64))

            for ann in self.img_annotations[image_id]:
                ann_cat = ann[AnnFileKeys.CATEGORY_ID]
                if ann_cat not in cat_ids:
                    continue
                cat_idx = cat_ids.index(ann_cat)

                ann_mask = self.prompts_processor.convert_mask(
                    ann[AnnFileKeys.SEGMENTATION], *img_size
                )
                ground_truths[i][ann_mask == 1] = cat_idx

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

        base_image_data = self.images[self.image_ids[idx]]
        image_ids, aux_cat_ids = self._extract_examples(
            base_image_data, num_examples, num_classes
        )

        if self.all_example_categories:
            aux_cat_ids = [aux_cat_ids[0]] + [
                set(self.img2cat[img]) for img in image_ids[1:]
            ]  # check if self.images must be called before

        cat_ids = sorted(list(set(itertools.chain(*aux_cat_ids))))
        cat_ids.insert(0, -1)  # add the background class

        # load, stack and preprocess the images
        images, image_key, ground_truths = self._get_images_or_embeddings(image_ids)

        # create the prompt dicts
        bboxes, masks, points, classes, img_sizes = self._get_prompts(
            image_ids, cat_ids, possible_prompt_types
        )

        # obtain padded tensors
        bboxes, flag_bboxes = utils.annotations_to_tensor(
            self.prompts_processor, bboxes, img_sizes, PromptType.BBOX
        )
        masks, flag_masks = utils.annotations_to_tensor(
            self.prompts_processor, masks, img_sizes, PromptType.MASK
        )
        points, flag_points = utils.annotations_to_tensor(
            self.prompts_processor, points, img_sizes, PromptType.POINT
        )

        # obtain ground truths
        if ground_truths is None:
            ground_truths = self.compute_ground_truths(image_ids, cat_ids)

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

        flag_examples = flags_merge(flag_masks, flag_points, flag_bboxes)

        return {
            image_key: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.PROMPT_POINTS: points,
            BatchKeys.FLAG_POINTS: flag_points,
            BatchKeys.PROMPT_BBOXES: bboxes,
            BatchKeys.FLAG_BBOXES: flag_bboxes,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.DIMS: dims,
            BatchKeys.CLASSES: classes,
            BatchKeys.IMAGE_IDS: image_ids,
            BatchKeys.GROUND_TRUTHS: ground_truths,
        }

    def __len__(self):
        return len(self.images)


class CocoLVISTestDataset(CocoLVISDataset, LabelAnythingTestDataset):
    def __init__(
        self,
        name,
        instances_path,  # Path
        img_dir=None,  # directory (only if images have to be loaded from disk)
        preprocess=ToTensor(),  # preprocess step
        emb_dir=None,
        seed=42,  # for reproducibility
        load_embeddings=None,
        load_gts=False,
        add_box_noise=False,
        dtype=torch.float32,
        support_params={},
    ):
        CocoLVISTestDataset.__init__(
            name=name,
            instances_path=instances_path,
            img_dir=img_dir,
            preprocess=preprocess,
            seed=seed,
            add_box_noise=add_box_noise,
            emb_dir=emb_dir,
            load_embeddings=load_embeddings,
            load_gts=load_gts,
            dtype=dtype,
        )
        LabelAnythingTestDataset.__init__()
        self.num_classes = len(list(self.cat2img.keys()))
        self.support_dataset = (
            CocoLVISDataset(preprocess=preprocess, **support_params),
        )

    def _extract_examples(self, cat2img: dict, img2cat: dict) -> list[int]:
        prompt_images = set()
        categories = list(self.categories.keys())
        random.shuffle(categories)
        for cat_id in categories:
            if cat_id not in cat2img:
                continue
            cat_images = cat2img[cat_id]
            _, img = max(map(lambda x: (len(img2cat[x]), x), cat_images))
            prompt_images.add(img)
        return prompt_images

    def _get_images_or_embeddings(self, image_ids):
        if self.load_embeddings:
            embeddings_gts = [self._load_safe(data) for data in image_ids]
            embeddings, gts = zip(*embeddings_gts)
            if not self.load_gts:
                gts = None
            return torch.stack(embeddings), BatchKeys.EMBEDDINGS, gts
        images = [
            self._load_and_preprocess_image(image_data) for image_data in image_ids
        ]
        gts = None
        if self.load_gts:
            gts = [self._load_safe(image_data)[1] for image_data in image_ids]
        return torch.stack(images), BatchKeys.IMAGES, gts

    def extract_prompts(
        self,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): # type: ignore
        cat2img = self.support_dataset.cat2img
        img2cat = self.support_dataset.img2cat
        img2cat_annotations = self.support_dataset.img2cat_annotations
        images = self.support_dataset.images

        image_ids = self._extract_examples(cat2img, img2cat)
        prompt_images = [images[x] for x in image_ids]

        prompt_images, prompt_images_key, _ = self._get_images_or_embeddings(
            prompt_images
        )

        cat_ids = sorted(list(self.categories.keys()))
        bboxes, masks, points, _, image_sizes = self._get_prompts(
            image_ids, cat_ids, images, img2cat_annotations
        )

        bboxes, flag_bboxes = utils.annotations_to_tensor(
            self.prompts_processor, bboxes, image_sizes, PromptType.BBOX
        )
        masks, flag_masks = utils.annotations_to_tensor(
            self.prompts_processor, masks, image_sizes, PromptType.MASK
        )
        points, flag_points = utils.annotations_to_tensor(
            self.prompts_processor, points, image_sizes, PromptType.POINT
        )
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

    def _get_prompts(
        self,
        image_ids: list,
        cat_ids: list,
        images: dict,
        img2cat_annotations: dict,
    ) -> (list, list, list, list, list): # type: ignore
        bboxes = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        masks = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]
        points = [{cat_id: [] for cat_id in cat_ids} for _ in image_ids]

        # get prompts from annotations
        classes = [list() for _ in image_ids]
        img_sizes = [
            (images[img_id]["height"], images[img_id]["width"]) for img_id in image_ids
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
                for ann, prompt_type in zip(
                    img2cat_annotations[img_id][cat_id], prompt_types
                ):
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
                                ann[AnnFileKeys.SEGMENTATION],
                                *img_size,
                            )
                        )
                    elif prompt_type == PromptType.POINT:
                        # take the point
                        mask = self.prompts_processor.convert_mask(
                            ann[AnnFileKeys.SEGMENTATION],
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
        image_id = self.image_ids[item]
        data, data_key, gt = self._get_images_or_embeddings([self.images[image_id]])
        cat_ids = sorted(list(self.cat2img.keys()))
        if gt is None:
            gt = self.compute_ground_truths([image_id], cat_ids)[0]
        else:
            gt = gt[0]
            # convert the ground truths to the right format
            ground_truths_copy = gt.clone()
            # set ground_truths to all 0s
            gt = torch.zeros_like(gt)
            for i, cat_id in enumerate(cat_ids):
                if cat_id == -1:
                    continue
                gt[ground_truths_copy == cat_id] = i

        dim = torch.as_tensor(gt.size())
        return {
            data_key: data,
            "dim": dim,
            "gt": gt,
        }

    def collate_fn(
        self, batched_input: list[dict[str, Any]]
    ) -> (dict[str, Any], torch.Tensor): # type: ignore
        data_key = "images" if "images" in batched_input[0].keys() else "embeddings"
        images = torch.stack([x[data_key] for x in batched_input])

        dims = torch.stack([x["dim"] for x in batched_input])

        max_dims = torch.max(dims, 0).values.tolist()
        gt = torch.stack([utils.collate_gts(x["gt"], max_dims) for x in batched_input])

        data_dict = {
            data_key: images,
            "dims": dims,
        }

        return data_dict, gt.long()


class LabelAnyThingOnlyImageDataset(Dataset):
    def __init__(self, directory=None, preprocess=None):
        super().__init__()
        self.directory = directory
        self.files = os.listdir(directory)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.directory, self.files[item])).convert("RGB")
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
