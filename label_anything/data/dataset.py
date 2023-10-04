import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import pandas as pd
import random
import torch
import itertools
from torchvision.transforms import Resize, ToTensor
import warnings

warnings.filterwarnings('always')


def compute_j_index(class_a, class_b):
    class_a = set(class_a)
    class_b = set(class_b)
    return len(class_a.intersection(class_b)) / len(class_a.union(class_b))


def convert_polygons(polygons):
    return [[(int(pol[i]), int(pol[i + 1])) for i in range(0, len(pol), 2)] for pol in polygons]


def get_mask(img_shape, reshape, segmentations):
    if segmentations == [[]]:  # for empty segmentation
        return reshape(torch.zeros(img_shape))[0]
    image = Image.new('L', img_shape[1:][::-1], 0)  # due to problem with shapes
    draw = ImageDraw.Draw(image)
    for pol in convert_polygons(segmentations):
        draw.polygon(pol, outline=1, fill=1)
    mask = np.asarray(image)
    mask = torch.Tensor(mask)
    return reshape(mask.unsqueeze(dim=0))


def get_mask_per_image(annotations, image_id, image_shape, reshape, target_classes):
    return torch.stack([
        get_mask(image_shape,
                 reshape,
                 itertools.chain(*annotations[(annotations.image_id == image_id) &
                                              (annotations.category_id == x)].segmentation.tolist()))
        for x in target_classes
    ])


def get_prompt_mask(annotations, image_shape, reshape, target_classes):
    return torch.stack([
        get_mask_per_image(annotations, x, image_shape, reshape, target_classes)
        for x in annotations.image_id.unique().tolist()
    ])


def get_bboxes(bboxes_entries, image_id, category_id, len_bbox):
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


def get_prompt_bbox_per_image(bbox_entries, img_id, target_classes, max_anns):
    res = [get_bboxes(bbox_entries, img_id, x, max_anns) for x in target_classes]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def get_prompt_bbox(bbox_entries, target_classes):
    max_anns = get_max_bbox(bbox_entries)
    res = [get_prompt_bbox_per_image(bbox_entries, x, target_classes, max_anns)
           for x in bbox_entries.image_id.unique().tolist()]
    return torch.stack([x[0] for x in res]), torch.stack([x[1] for x in res])


def get_max_bbox(annotations):
    return max(
        len(annotations[(annotations.image_id == img) & (annotations.category_id == cat)])
        for img in annotations.image_id.unique() for cat in annotations.category_id.unique()
    )


def get_gt(annotations, image_shape, target_classes):
    gt = Image.new('L', image_shape[1:][::-1], 0)
    draw = ImageDraw.Draw(gt)
    for ix, c in enumerate(target_classes, start=1):
        polygons = convert_polygons(itertools.chain(*annotations[annotations.category_id == c].segmentation.tolist()))
        for pol in polygons:
            draw.polygon(pol, outline=1, fill=ix)
    gt = np.asarray(gt)
    gt = torch.Tensor(gt)
    return gt


class LabelAnythingDataset(Dataset):
    def __init__(
            self,
            instances,  #data
            directory=None,  # directory (only if images have to be loaded from disk)
            num_max_examples=10,  # number of max examples to be given for the target image
            preprocess=ToTensor(),  # preprocess step
            j_index_value=.5,  # threshold for extracting examples
            seed=42,  # for reproducibility
            resize_dim=224,  # shape for stacking images
    ):
        super().__init__()
        self.annotations = pd.DataFrame(instances['annotations'])
        self.images = pd.DataFrame(instances['images'])
        self.load_from_dir = directory is not None
        self.categories = pd.DataFrame(instances['categories'])
        self.num_max_examples = num_max_examples
        self.directory = directory
        self.preprocess = preprocess
        self.j_index_value = j_index_value
        self.seed = seed
        self.resize = Resize((resize_dim, resize_dim))
        self.img_shape = (1, resize_dim, resize_dim)

    def __load_image(self, img):
        if self.load_from_dir:
            return Image.open(f'{self.directory}/{img["id"]}.jpg')
        return Image.open(BytesIO(requests.get(img["coco_url"]).content))

    def __extract_examples(self, img_id):
        target_classes = self.annotations[self.annotations.image_id == img_id]['category_id'].tolist()
        class_projection = self.annotations[['image_id', 'category_id']].groupby(by='image_id')['category_id'].apply(list).reset_index(name='category_id')
        class_projection['j_score'] = class_projection.apply(lambda x: compute_j_index(target_classes, x.category_id),
                                                             axis=1)
        class_projection = class_projection[class_projection['j_score'] > self.j_index_value]
        num_samples = random.randint(1, self.num_max_examples)
        print(f'extracting {num_samples} examples')
        return class_projection.sample(n=num_samples, replace=True,
                                       random_state=self.seed).image_id.tolist(), list(set(target_classes))

    def __getitem__(self, item):
        image_id = self.images.iloc[item]  # image row id
        target = self.__load_image(image_id)  # load image
        target = target if not self.preprocess else self.preprocess(target)  # preprocess

        # choose similar content with respect to target image
        example_ids, classes = self.__extract_examples(image_id['id'])
        examples = [self.__load_image(row)
                    for ix, row in self.images[self.images.id.isin(example_ids)].iterrows()]  # load images
        # annotations useful for prompt
        prompt_annotations = self.annotations[(self.annotations.image_id.isin(example_ids)) &
                                              (self.annotations.category_id.isin(classes))]

        # bboxes
        bbox_annotations = prompt_annotations[['image_id', 'bbox', 'category_id']]
        prompt_bbox, flag_bbox = get_prompt_bbox(bbox_annotations, classes)

        # masks
        mask_annotations = prompt_annotations[['image_id', 'segmentation', 'category_id']]
        prompt_mask = get_prompt_mask(mask_annotations, target.shape, self.resize, classes)

        # gt
        target_annotations = self.annotations[self.annotations.image_id == image_id['id']][['category_id', 'segmentation']]
        gt = self.resize(get_gt(target_annotations, target.shape, classes).unsqueeze(0)).squeeze(0)

        if self.preprocess:
            examples = [self.resize(self.preprocess(x)) for x in examples]
        examples = torch.stack(examples)  # load and stack images

        target = self.resize(target)

        return {
            'target': target,  # 3 x h x w
            'examples': examples,  # n x 3 x h x w
            'prompt_mask': prompt_mask,  # n x c x h x w
            #'prompt_point': None,
            'prompt_bbox': prompt_bbox,  # n x c x m x 4
            'flag_bbox': flag_bbox,  # n x c x m
            'gt': gt,  # h x w
            'classes': {
                ix: c for ix, c in enumerate(classes, start=1)
            }
        }

    def __len__(self):
        return len(self.images)


# main for testing the class
if __name__ == '__main__':
    import json
    from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

    with open('lvis_v1_train.json') as f:
        instances = json.load(f)

    preprocess = Compose([
        ToTensor(),
        #Resize((1000, 1000)),
    ])

    dataset = LabelAnythingDataset(
        instances=instances,
        preprocess=preprocess,
        num_max_examples=10,
        j_index_value=.1,
    )

    out = dataset[0]

    for k, v in out.items():
        print(f'{k}: {v.size()}' if isinstance(v, torch.Tensor) else f'{k}: {v}')