import itertools
import os
import random
import warnings
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
import torchvision.transforms
import label_anything.data.utils as utils
from label_anything.data.examples import ExampleGeneratorPowerLawUniform
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    PILToTensor,
    ToTensor,
)
from label_anything.data.transforms import CustomNormalize, CustomResize, PromptsProcessor
from safetensors import safe_open


warnings.filterwarnings("ignore")


class PromptType(Enum):
    BBOX = 0
    MASK = 1
    POINT = 2


class LabelAnythingDataset(Dataset):
    def __init__(
        self,
        instances_path,  # Path
        img_dir=None,  # directory (only if images have to be loaded from disk)
        max_num_examples=10,  # number of max examples to be given for the target image
        preprocess=ToTensor(),  # preprocess step
        seed=42,  # for reproducibility
        emb_dir=None,
    ):
        super().__init__()
        instances = utils.load_instances(instances_path)
        self.emb_dir = emb_dir
        self.load_embeddings = self.emb_dir is not None
        self.load_from_dir = img_dir is not None
        self.img_dir = img_dir
        assert not (self.load_from_dir and self.load_embeddings)

        # id to annotation
        self.annotations = {x["id"]: x for x in instances["annotations"]}
        # id to category
        self.categories = {x["id"]: x for x in instances["categories"]}
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

        # id to image
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
        self.__set_all_seeds()
        self.reset_num_examples()

    def __set_all_seeds(self):
        """Enable reproducibility.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.log_images = False

    def reset_num_examples(self):
        """Set the number of examples for the next query image.
        """
        self.num_examples = random.randint(1, self.max_num_examples)

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
        with safe_open(f"{self.emb_dir}/{img_data['id']}.safetensors", framework="pt") as f:
            tensor = f.get_slice("embedding")
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
        return Image.open(BytesIO(requests.get(img_data["coco_url"]).content)).convert("RGB")

    def _extract_examples(self, img_data: dict) -> (list, list):
        """Chooses examples (and categories) for the query image.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            (list, list): Returns two lists:
                1. examples: A list of image ids of the examples.
                2. cats: A list of sets of category ids of the examples.
        """
        return self.example_generator.generate_examples(
            query_image_id=img_data["id"],
            image_classes=self.img2cat[img_data["id"]],
            num_examples=self.num_examples,
        )

    def _get_annotations(self, image_ids, cat_ids):
        bboxes = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}
        masks = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}
        points = {img_id: {cat_id: [] for cat_id in cat_ids} for img_id in image_ids}

        # get prompts from annotations
        classes = {img_id: set() for img_id in image_ids}

        for img_id in image_ids:
            img_size = (self.images[img_id]["height"], self.images[img_id]["width"])
            for cat_id in cat_ids:
                # for each pair (image img_id and category cat_id)
                if cat_id not in self.img2cat_annotations[img_id]:
                    # the chosen category is not in the iamge
                    continue
                
                classes[img_id].add(cat_id)
                for ann in self.img2cat_annotations[img_id][cat_id]:
                    # choose the prompt type
                    prompt_type = random.choice(list(PromptType))

                    if prompt_type == PromptType.BBOX:
                        # take the bbox
                        bboxes[img_id][cat_id].append(
                            self.prompts_processor.convert_bbox(
                                ann["bbox"],
                                *img_size,
                                noise=True,),
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
        # TODO: remove this
        return torch.rand(len(image_ids), 256, 64, 64), "embeddings"
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
        base_image_data = self.images[self.image_ids[item]]

        image_ids, aux_cat_ids = self._extract_examples(base_image_data)
        cat_ids = list(set(itertools.chain(*aux_cat_ids)))
        cat_ids.insert(0, -1)  # add the background class

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

        data_dict =  {
            image_key: images,
            "prompt_masks": masks,
            "flag_masks": flag_masks,
            "prompt_points": points,
            "flag_points": flag_points,
            "prompt_bboxes": bboxes,
            "flag_bboxes": flag_bboxes,
            "dims": dims,
            "classes": list(map(list, classes.values())),
            "ground_truths": ground_truths,
        }

        if self.log_images and self.load_embeddings:
            log_images =[
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
                ground_truths[img_id][cat_id] = np.zeros(img_size, dtype=np.uint8)
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
                        ground_truths[img_id][cat_id].astype(np.uint8)
                        for cat_id in cat_ids
                    ]
                )
            )
            # add a zeroes tensor to the first dimension
            ground_truth = torch.cat(
                [
                    torch.zeros((1, *ground_truth.shape[1:])).type(torch.uint8),
                    ground_truth,
                ]
            )
            ground_truths[img_id] = torch.argmax(ground_truth, 0)

        return list(ground_truths.values())

    def __len__(self):
        return len(self.images)

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

    def collate_fn(
        self, batched_input: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Performs the collate_fn, which is useful for batching data points in a dataloader.

        Arguments:
            batched_input: list of batch_size elements, in which each element is a dict with the following entries:
                'target': query image as a torch tensor of shape 3 x H x W.
                'examples': example image as a torch tensor of shape M x 3 x H x W, where M is the number of examples
                    extracted for the given query image.
                'prompt_mask': example image masks as a torch tensor of shape M x C x H x W, where M is the number of
                    examples extracted for the given query image and C is the number of classed associated to it.
                'prompt_coords': example image coordinates as a torch tensor of shape M x C x N x K x 2, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, N is the maximum number of annotations associated to a pair (image, class), and K is
                    the number of points extracted.
                'flag_coords': example image coordinate flags as a torch tensor of shape M x C x N x K, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, N is the maximum number of annotations associated to a pair (image, class), and K is
                    the number of points extracted.
                'prompt_bbox': example image bounding boxes as a torch tensor of shape M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, and N is the maximum number of annotations associated to a pair (image, class). The
                    last dimension is 4 because a single bounding box is represented by the top-left and bottom-right
                    coordinates.
                'flag_bbox': example image bounding box flags as a torch tensor of shape M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, and N is the maximum number of annotations associated to a pair (image, class).
                'gt': query image classes mask as a tensor of shape H x W, in which each pixel has a certain value k if
                    that pixel is in the mask of the k-th class associated to the query image.
                'classes': dict in which each pair k: v is ith class corresponding to class id.

        Returns:
            Dict[str, Any]: batched dictionary having the following entries:
                'query_image': query image as a torch tensor of shape B x 3 x H x W.
                'example_images': example images as a torch tensor of shape B x M x 3 x H x W.
                'point_coords':  example image coordinates as a torch tensor of shape B x M x C x N x K x 2, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated to
                    the given image, N is the maximum number of annotations associated to a pair (image, class), and K
                    is the number of points extracted.
                'point_flags': example image coordinate flags as a torch tensor of shape B xM x C x N x K, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated to
                    the given image, N is the maximum number of annotations associated to a pair (image, class), and K
                    is the number of points extracted.
                'boxes': example image bounding boxes as a torch tensor of shape B x M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, and N is the maximum number of annotations associated to a pair (image, class). The
                    last dimension is 4 because a single bounding box is represented by the top-left and bottom-right
                    coordinates.
                'box_flags': example image bounding box flags as a torch tensor of shape B x M x C x N x 4, where M is
                    the number of examples extracted for the given query image, C is the number of classes associated to
                    the given image, and N is the maximum number of annotations associated to a pair (image, class).
                'mask_inputs': example image masks as a torch tensor of shape B x M x C x H x W, where M is the number
                    of examples extracted for the given query image and C is the number of classed associated to it.
            torch.Tensor: batched output masks as a torch tensor of shape B x H x W.

        """
        # classes
        max_classes = max([x["prompt_masks"].size(1) for x in batched_input])

        # gt
        dims = torch.stack([x["dims"] for x in batched_input])
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        ground_truths = [x["ground_truths"] for x in batched_input]
        ground_truths = torch.stack(
            [utils.collate_batch_gts(x, max_dims) for x in ground_truths]
        )

        # prompt mask
        masks = [x["prompt_masks"] for x in batched_input]
        flags = [x["flag_masks"] for x in batched_input]
        masks_flags = [
            utils.collate_mask(m, f, max_classes) for (m, f) in zip(masks, flags)
        ]
        masks = torch.stack([x[0] for x in masks_flags])
        flag_masks = torch.stack([x[1] for x in masks_flags])

        # prompt bbox
        bboxes = [x["prompt_bboxes"] for x in batched_input]
        flags = [x["flag_bboxes"] for x in batched_input]
        max_annotations = max(x.size(2) for x in bboxes)
        bboxes_flags = [
            utils.collate_bbox(bboxes[i], flags[i], max_classes, max_annotations)
            for i in range(len(bboxes))
        ]
        bboxes = torch.stack([x[0] for x in bboxes_flags])
        flag_bboxes = torch.stack([x[1] for x in bboxes_flags])

        # prompt coords
        points = [x["prompt_points"] for x in batched_input]
        flags = [x["flag_points"] for x in batched_input]
        max_annotations = max(x.size(2) for x in points)
        points_flags = [
            utils.collate_coords(points[i], flags[i], max_classes, max_annotations)
            for i in range(len(points))
        ]
        points = torch.stack([x[0] for x in points_flags])
        flag_points = torch.stack([x[1] for x in points_flags])

        # aux gts
        classes = [x["classes"] for x in batched_input]

        # images
        if "embeddings" in batched_input[0].keys():
            image_key = "embeddings"
            images = torch.stack([x[image_key] for x in batched_input])
        else:
            image_key = "images"
            images = torch.stack([torch.stack(x["images"]) for x in batched_input])

        data_dict = {
            image_key: images,
            "prompt_points": points,
            "flag_points": flag_points,
            "prompt_bboxes": bboxes,
            "flag_bboxes": flag_bboxes,
            "prompt_masks": masks,
            "flag_masks": flag_masks,
            "dims": dims,
            "classes": classes,
        }

        if self.log_images and not self.load_embeddings:
            log_images = torch.stack([x["images"] for x in batched_input])
            data_dict["images"] = log_images

        # reset dataset parameters
        self.reset_num_examples()

        return data_dict, ground_truths


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
    dataset = LabelAnythingDataset(
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
