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
from label_anything.data.utils import AnnFileKeys, BatchKeys, PromptType

warnings.filterwarnings("ignore")


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
        preprocess=ToTensor(),
        seed: int = 42,
        load_gts: bool = False,
        do_subsample: bool = True,
        add_box_noise: bool = True,
        prompt_types: list[PromptType] = [
            PromptType.BBOX,
            PromptType.MASK,
            PromptType.POINT,
        ],
    ):
        """Initialize the dataset.

        Args:
            name (str): A name for the dataset (e.g. "coco", "lvis").
            instances_path (str): Path to the instances json file.
            img_dir (Optional[str], optional): Path to the directory containing the images. Defaults to None.
            emb_dir (Optional[str], optional): Path to the directory containing the embeddings. Defaults to None.
            max_num_examples (int, optional): Maximum number of examples for each image. Defaults to 10.
            max_points_annotations (int, optional): Maximum number of sparse prompts. Defaults to 50.
            preprocess (_type_, optional): A preprocessing step to apply to the images. Defaults to ToTensor().
            seed (int, optional): For reproducibility. Defaults to 42.
            load_gts (bool, optional): Specify if ground truth masks are precomputed. Defaults to False.
            do_subsample (bool, optional): Specify if classes should be randomly subsampled. Defaults to True.
            add_box_noise (bool, optional): Add noise to the boxes (useful for training). Defaults to True.
            prompt_types (list[PromptType], optional): List of prompt types to be used. Defaults to [PromptType.BBOX, PromptType.MASK, PromptType.POINT].
        """
        super().__init__()
        print(f"Loading dataset annotations from {instances_path}...")

        assert (
            img_dir is not None or emb_dir is not None
        ), "Either img_dir or emb_dir must be provided."
        assert (
            not load_gts or emb_dir is not None
        ), "If load_gts is True, emb_dir must be provided."
        assert len(prompt_types) > 0, "prompt_types must be a non-empty list."

        self.name = name

        self.img_dir = img_dir
        self.emb_dir = emb_dir
        self.load_gts = load_gts
        self.max_points_per_annotation = max_points_per_annotation
        self.max_points_annotations = max_points_annotations
        self.do_subsample = do_subsample
        self.add_box_noise = add_box_noise
        self.prompt_types = prompt_types

        # seeds
        self.reset_seed(seed)

        # load instances
        instances = utils.load_instances(instances_path)
        self.annotations = {
            x[AnnFileKeys.ID]: x for x in instances[AnnFileKeys.ANNOTATIONS]
        }
        self.categories = {
            x[AnnFileKeys.ID]: x for x in instances[AnnFileKeys.CATEGORIES]
        }

        # useful dicts
        (
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

        # example generator/selector
        self.example_generator = ExampleGeneratorPowerLawUniform(
            categories_to_imgs=self.cat2img, generator=self.torch_rng
        )

        # processing
        self.preprocess = preprocess
        self.prompts_processor = PromptsProcessor(
            long_side_length=1024, masks_side_length=256, np_rng=self.np_rng
        )

    def reset_seed(self, seed: int) -> None:
        """Reset the seed of the dataset.

        Args:
            seed (int): The new seed.
        """
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.torch_rng = torch.Generator().manual_seed(self.seed)
        if hasattr(self, "example_generator"):
            self.example_generator.generator = self.torch_rng
        if hasattr(self, "prompts_processor"):
            self.prompts_processor.np_rng = self.np_rng

    def _load_annotation_dicts(self) -> (dict, dict, dict, dict):
        """Load useful annotation dicts.

        Returns:
            (dict, dict, dict, dict): Returns four dictionaries:
                1. img2cat: A dictionary mapping image ids to sets of category ids.
                2. img2cat_annotations: A dictionary mapping image ids to dictionaries mapping category ids to annotations.
                3. cat2img: A dictionary mapping category ids to sets of image ids.
                4. cat2img_annotations: A dictionary mapping category ids to dictionaries mapping image ids to annotations.
        """
        img2cat_annotations = {}
        cat2img_annotations = {}

        img2cat = {}
        cat2img = {}

        for ann in self.annotations.values():
            if AnnFileKeys.ISCROWD in ann and ann[AnnFileKeys.ISCROWD] == 1:
                continue

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
        return img2cat, img2cat_annotations, cat2img, cat2img_annotations

    def _load_safe(self, img_data: dict) -> (torch.Tensor, Optional[torch.Tensor]):
        """Open a safetensors file and load the embedding and the ground truth.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            (torch.Tensor, Optional[torch.Tensor]): Returns a tuple containing the embedding and the ground truth.
        """
        assert self.emb_dir is not None, "emb_dir must be provided."
        gt = None

        f = load_file(
            f"{self.emb_dir}/{str(img_data[AnnFileKeys.ID]).zfill(12)}.safetensors"
        )
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
        if self.img_dir is not None:
            return Image.open(
                f'{self.img_dir}/{img_data["coco_url"].split("/")[-1]}'
            ).convert("RGB")
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
        return image if not self.preprocess else self.preprocess(image)

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

    def _extract_examples(
        self, img_data: dict, num_examples: int
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
            self.np_rng.poisson(poisson_mean) + 1, 1, self.max_points_per_annotation
        )

    def _get_prompts(
        self, image_ids: list, cat_ids: list
    ) -> (list, list, list, list, list):
        """Get the annotations for the chosen examples.

        Args:
            image_ids (list): A list of image ids of the examples.
            cat_ids (list): A list of sets of category ids of the examples.

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
                    prompt_types = self.rng.choices(self.prompt_types, k=n_ann)

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
    ) -> (torch.Tensor, str, Optional[torch.Tensor]):
        """Load, stack and preprocess the images or the embeddings.

        Args:
            image_ids (list[int]): A list of image ids.

        Returns:
            (torch.Tensor, str, Optional[torch.Tensor]): Returns a tuple containing the images or the embeddings, the key of the returned tensor and the ground truths.
        """
        if self.emb_dir is not None:
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
        ground_truths = [dict() for _ in range (len(image_ids))]
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

    def annotations_to_tensor(
        self, annotations: list, img_sizes: list, prompt_type: PromptType
    ) -> torch.Tensor:
        """Convert a list of annotations to a tensor.

        Args:
            annotations (list): A list of annotations.
            img_sizes (list): A list of tuples containing the image sizes.
            prompt_type (PromptType): The type of the prompt.

        Returns:
            torch.Tensor: The tensor containing the annotations.
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

    def __getitem__(self, idx_num_examples: tuple[int, int]) -> dict:
        """Get an item from the dataset.

        Args:
            idx_num_examples (tuple[int, int]): A tuple containing the index of the image and the number of examples to be chosen.

        Returns:
            dict: A dictionary containing the data.
        """
        idx, num_examples = idx_num_examples
        base_image_data = self.images[self.image_ids[idx]]
        image_ids, aux_cat_ids = self._extract_examples(base_image_data, num_examples)
        cat_ids = list(set(itertools.chain(*aux_cat_ids)))
        cat_ids.insert(0, -1)  # add the background class

        # load, stack and preprocess the images
        images, image_key, ground_truths = self._get_images_or_embeddings(image_ids)

        # create the prompt dicts
        bboxes, masks, points, classes, img_sizes = self._get_prompts(
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

        data_dict = {
            image_key: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.PROMPT_POINTS: points,
            BatchKeys.FLAG_POINTS: flag_points,
            BatchKeys.PROMPT_BBOXES: bboxes,
            BatchKeys.FLAG_BBOXES: flag_bboxes,
            BatchKeys.DIMS: dims,
            BatchKeys.CLASSES: classes,
            BatchKeys.IMAGE_IDS: image_ids,
            BatchKeys.GROUND_TRUTHS: ground_truths,
        }

        return data_dict

    def __len__(self):
        return 5
        return len(self.images)


class CocoLVISTestDataset(CocoLVISDataset):
    def __init__(
        self,
        name,
        instances_path,  # Path
        img_dir=None,  # directory (only if images have to be loaded from disk)
        preprocess=ToTensor(),  # preprocess step
        emb_dir=None,
        seed=42,  # for reproducibility
        load_gts=False,
        add_box_noise=False,
    ):
        super(CocoLVISTestDataset, self).__init__(
            name=name,
            instances_path=instances_path,
            img_dir=img_dir,
            preprocess=preprocess,
            seed=seed,
            add_box_noise=add_box_noise,
            emb_dir=emb_dir,
            load_gts=load_gts,
        )
        self.num_classes = len(list(self.cat2img.keys()))

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
        if self.emb_dir is not None:
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
        cat2img: dict,
        img2cat: dict,
        images: dict,
        img2cat_annotations: dict,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        image_ids = self._extract_examples(cat2img, img2cat)
        prompt_images = [images[x] for x in image_ids]

        prompt_images, prompt_images_key, _ = self._get_images_or_embeddings(
            prompt_images
        )

        cat_ids = list(self.categories.keys())
        bboxes, masks, points, _, image_sizes = self._get_prompts(
            image_ids, cat_ids, images, img2cat_annotations
        )

        bboxes, flag_bboxes = self.annotations_to_tensor(
            bboxes, image_sizes, PromptType.BBOX
        )
        masks, flag_masks = self.annotations_to_tensor(
            masks, image_sizes, PromptType.MASK
        )
        points, flag_points = self.annotations_to_tensor(
            points, image_sizes, PromptType.POINT
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
    ) -> (list, list, list, list, list):
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
        image_id = self.image_ids[item]
        data, data_key, gt = self._get_images_or_embeddings([self.images[image_id]])
        cat_ids = list(self.cat2img.keys())
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
        data_dict = {
            data_key: data,
            "dim": dim,
            "gt": gt,
        }
        return data_dict

    def collate_fn(
        self, batched_input: list[dict[str, Any]]
    ) -> (dict[str, Any], torch.Tensor):
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
