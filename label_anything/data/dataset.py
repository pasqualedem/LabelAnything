import os
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import random
import torch
from torchvision.transforms import Resize, ToTensor, InterpolationMode
import warnings

from label_anything.data import utils


warnings.filterwarnings("ignore")


class LabelAnythingDataset(Dataset):
    def __init__(
        self,
        instances_path,  # Path
        directory=None,  # directory (only if images have to be loaded from disk)
        num_max_examples=10,  # number of max examples to be given for the target image
        preprocess=ToTensor(),  # preprocess step
        j_index_value=0.5,  # threshold for extracting examples
        seed=42,  # for reproducibility
        resize_dim=224,  # shape for stacking images
        max_mum_coords=10,  # max number of coords for each example for each class
    ):
        super().__init__()
        instances = utils.load_instances(instances_path)
        self.annotations = pd.DataFrame(instances["annotations"])
        self.images = pd.DataFrame(instances["images"])
        self.load_from_dir = directory is not None
        self.categories = pd.DataFrame(instances["categories"])
        self.num_max_examples = num_max_examples
        self.directory = directory
        self.preprocess = preprocess
        self.j_index_value = j_index_value
        self.seed = seed
        self.resize = Resize((resize_dim, resize_dim))
        self.resize_mask = Resize(
            (resize_dim, resize_dim), interpolation=InterpolationMode.NEAREST
        )
        self.img_shape = (1, resize_dim, resize_dim)
        self.max_num_coords = max_mum_coords
        assert (
            self.max_num_coords > 1
        ), "maximum number of coords must be greater then 1."
        self.num_coords = random.randint(1, max_mum_coords)
        self.num_examples = random.randint(1, self.num_max_examples)

    def __load_image(self, img):
        if self.load_from_dir:
            return Image.open(f'{self.directory}/{img["id"]}.jpg')
        return Image.open(BytesIO(requests.get(img["coco_url"]).content))

    def __extract_examples(self, img_id):
        target_classes = self.annotations[self.annotations.image_id == img_id][
            "category_id"
        ].tolist()
        class_projection = (
            self.annotations[["image_id", "category_id"]]
            .groupby(by="image_id")["category_id"]
            .apply(list)
            .reset_index(name="category_id")
        )
        class_projection["j_score"] = class_projection.apply(
            lambda x: utils.compute_j_index(target_classes, x.category_id), axis=1
        )
        class_projection = class_projection[
            class_projection["j_score"] > self.j_index_value
        ]
        print(f"extracting {self.num_examples} examples")
        return class_projection.sample(
            n=self.num_examples, replace=True, random_state=self.seed
        ).image_id.tolist(), list(set(target_classes))

    def __getitem__(self, item):
        image_id = self.images.iloc[item]  # image row id
        target = self.__load_image(image_id)  # load image
        target = (
            target if not self.preprocess else self.preprocess(target)
        )  # preprocess

        # choose similar content with respect to target image
        example_ids, classes = self.__extract_examples(image_id["id"])
        examples = [
            self.__load_image(row)
            for ix, row in self.images[self.images.id.isin(example_ids)].iterrows()
        ]  # load images
        # annotations useful for prompt
        prompt_annotations = self.annotations[
            (self.annotations.image_id.isin(example_ids))
            & (self.annotations.category_id.isin(classes))
        ]

        # bboxes
        bbox_annotations = prompt_annotations[["image_id", "bbox", "category_id"]]
        prompt_bbox, flag_bbox = utils.get_prompt_bbox(bbox_annotations, classes)

        # masks
        mask_annotations = prompt_annotations[
            ["image_id", "segmentation", "category_id"]
        ]
        prompt_mask = utils.get_prompt_mask(
            mask_annotations, target.shape, self.resize_mask, classes
        ).squeeze(dim=2)

        # gt
        target_annotations = self.annotations[
            self.annotations.image_id == image_id["id"]
        ][["category_id", "segmentation"]]
        gt = self.resize_mask(
            utils.get_gt(target_annotations, target.shape, classes).unsqueeze(0)
        ).squeeze(0)

        if self.preprocess:
            examples = [self.resize(self.preprocess(x)) for x in examples]
        examples = torch.stack(examples)  # load and stack images

        prompt_coords, flag_coords = utils.get_prompt_coords(
            annotations=mask_annotations,
            target_classes=classes,
            num_coords=self.num_coords,
            original_shape=target.shape,
            resize=self.resize_mask,
        )

        target = self.resize(target)

        return {
            "target": target,  # 3 x h x w
            "examples": examples,  # n x 3 x h x w
            "prompt_mask": prompt_mask,  # n x c x h x w
            "prompt_coords": prompt_coords,
            "flag_coords": flag_coords,
            "prompt_bbox": prompt_bbox,  # n x c x m x 4
            "flag_bbox": flag_bbox,  # n x c x m
            "gt": gt,  # h x w
            "classes": {ix: c for ix, c in enumerate(classes, start=1)},
        }

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batched_input):
        classes = [x["classes"] for x in batched_input]
        new_classes = utils.rearrange_classes(classes)
        gts = [x["gt"] for x in batched_input]
        gts = torch.stack(
            [utils.collate_gt(x, classes[ix], new_classes) for ix, x in enumerate(gts)]
        )
        masks = [x["prompt_mask"] for x in batched_input]
        masks = torch.stack(
            [
                utils.collate_mask(mask, classes[i], new_classes)
                for i, mask in enumerate(masks)
            ]
        )
        bboxes = [x["prompt_bbox"] for x in batched_input]
        flags = [x["flag_bbox"] for x in batched_input]
        max_annotations = max(x.size(2) for x in bboxes)
        bboxes_flags = [
            utils.collate_bbox(
                bboxes[i], flags[i], classes[i], new_classes, max_annotations
            )
            for i in range(len(bboxes))
        ]
        bboxes = torch.stack([x[0] for x in bboxes_flags])
        bbox_flags = torch.stack([x[1] for x in bboxes_flags])
        coords = [x["prompt_coords"] for x in batched_input]
        flags = [x["flag_coords"] for x in batched_input]
        coords_flags = [
            utils.collate_coords(
                coords[i], flags[i], classes[i], new_classes, max_annotations
            )
            for i in range(len(coords))
        ]
        coords = torch.stack([x[0] for x in coords_flags])
        coord_flags = torch.stack([x[1] for x in coords_flags])
        query_image = torch.stack([x["target"] for x in batched_input])
        example_images = torch.stack([x["examples"] for x in batched_input])

        data_dict = {
            "query_image": query_image,
            "example_images": example_images,
            "point_coords": coords,
            "point_flags": coord_flags,
            "boxes": bboxes,
            "box_flags": bbox_flags,
            "mask_inputs": masks,
        }

        return data_dict, gts


class LabelAnyThingOnlyImageDataset(Dataset):
    def __init__(
            self,
            directory=None,
            preprocess=None
    ):
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
    from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
    from torch.utils.data import DataLoader

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
    data_dict, gt = next(iter(dataloader))

    print([f"{k}: {v.size()}" for k, v in data_dict.items()])
    print(f"gt: {gt.size()}")
