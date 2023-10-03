import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import pandas as pd
import random
import torch


def __compute_j_index(class_a, class_b):
    class_a = set(class_a)
    class_b = set(class_b)
    return len(class_a.intersection(class_b)) / len(class_a.union(class_b))


def __convert_polygons(polygons):
    ans = []
    for pol in polygons:
        ans += [[(int(pol[i]), int(pol[i + 1])) for i in range(0, len(pol), 2)]]
    return ans


def __apply_mask(img, segmentations):
    image = Image.new('L', img.shape[1:][::-1], 0)  # due to problem with shapes
    draw = ImageDraw.Draw(image)
    for pol in __convert_polygons(segmentations):
        draw.polygon(pol, outline=1, fill=1)
    mask = np.asarray(image)
    mask = torch.Tensor(mask)
    return mask




def __get_bboxes(bboxes_entries, image_id, category_id, len_bbox):
    bbox = torch.Tensor(
        bboxes_entries[(bboxes_entries.image_id == image_id) & (bboxes_entries.category_id == category_id)]['bbox'].tolist()
    )
    ans = torch.cat([
        bbox, torch.zeros((len_bbox - bbox.size(0), 4))
    ], dim=0)
    if bbox.size(0) == 0:
        flag = torch.full((len_bbox, ), fill_value=False)
    elif bbox.size(0) == len_bbox:
        flag = torch.full((len_bbox, ), fill_value=True)
    else:
        flag = torch.cat(
            [torch.full((bbox.size(0), ), fill_value=True), torch.full(((len_bbox - bbox.size(0)), ), fill_value=False)]
        )
    return ans, flag


def __get_prompt_bbox_per_image(bbox_entries, img_id, target_classes, max_anns):
    res = [__get_bboxes(bbox_entries, img_id, x, max_anns) for x in target_classes]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def __get_prompt_bbox(bbox_entries, target_classes):
    max_anns = __get_max_bbox(bbox_entries)
    res = [__get_prompt_bbox_per_image(bbox_entries, x, target_classes, max_anns)
           for x in bbox_entries.image_id.unique().tolist()]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def __get_max_bbox(annotations):
    return max(
        len(annotations[(annotations.image_id == img) & (annotations.category_id == cat)])
        for img in annotations.image_id.unique() for cat in annotations.category_id.unique()
    )


class LabelAnythingDataset(Dataset):
    def __init__(
            self,
            instances,
            directory=None,
            num_max_examples=10,
            preprocess=None,
            j_index_value=.2,
            seed=42,
    ):
        self.annotations = pd.DataFrame(instances['annotations'])
        self.images = pd.DataFrame(instances['images'])
        self.load_from_dir = directory is not None
        self.categories = pd.DataFrame(instances['categories'])
        self.num_max_examples = num_max_examples
        self.directory = directory
        self.preprocess = preprocess
        self.j_index_value = j_index_value
        self.seed = seed

    def __load_image(self, img):
        if self.load_from_dir:
            return Image.open(f'{self.directory}/{img["id"]}.jpg')
        return Image.open(BytesIO(requests.get(img["coco_url"]).content))

    def __extract_examples(self, img_id):
        target_classes = self.annotations[self.annotations.image_id == img_id]['category_id'].tolist()
        class_projection = self.annotations.groupby('image_id')['category_id'].apply(list).reset_index(name='image_id')
        class_projection['j_score'] = class_projection.apply(lambda x: __compute_j_index(target_classes, x.category_id),
                                                             axis=1)
        class_projection = class_projection[class_projection['j_score'] > self.j_index_value]
        num_samples = random.randint(1, self.num_max_examples)
        return class_projection.sample(n=num_samples, replace=False,
                                       random_state=self.seed).image_id.tolist(), target_classes

    def __getitem__(self, item):
        image_id = self.images.iloc[item]  # image id

        target = self.__load_image(image_id)  # load image
        target = target if not self.preprocess else self.preprocess(target)  # preprocess

        example_ids, classes = self.__extract_examples(image_id)  # choose similar content with respect to target image
        examples = [self.__load_image(x) for x in example_ids]  # load images
        # annotations useful for prompt
        prompt_annotations = self.annotations[(self.annotations.image_id.isin(example_ids)) &
                                              (self.annotations.category_id.isin(classes))]

        # bboxes
        bbox_annotations = prompt_annotations[['image_id', 'bbox', 'category_id']]
        prompt_bbox = __get_prompt_bbox(bbox_annotations, classes)

        if self.preprocess:
            examples = [self.preprocess(x) for x in examples]
        examples = torch.stack(examples)  # load and stack images

        return {
            'target': target,
            'examples': examples,
            'prompt_mask': None,
            'prompt_point': None,
            'prompt_bbox': prompt_bbox,
            'gt': None,
        }

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import json
    with open('lvis_v1_train.json') as f:
        instances = json.load(f)

    annotations = pd.DataFrame(instances['annotations'])
    example = annotations[annotations.image_id == 195042]
    print(__get_prompt_bbox(example, target_classes=[811, 430, 431]))

