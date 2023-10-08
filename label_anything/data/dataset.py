from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import random
import torch
from torchvision.transforms import Resize, ToTensor
import warnings
import utils

warnings.filterwarnings('ignore')


class LabelAnythingDataset(Dataset):
    def __init__(
            self,
            instances_path,  #Path
            directory=None,  # directory (only if images have to be loaded from disk)
            num_max_examples=10,  # number of max examples to be given for the target image
            preprocess=ToTensor(),  # preprocess step
            j_index_value=.5,  # threshold for extracting examples
            seed=42,  # for reproducibility
            resize_dim=224,  # shape for stacking images
    ):
        super().__init__()
        instances = utils.load_instances(instances_path)
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
        class_projection['j_score'] = class_projection.apply(lambda x: utils.compute_j_index(target_classes, x.category_id),
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
        prompt_bbox, flag_bbox = utils.get_prompt_bbox(bbox_annotations, classes)

        # masks
        mask_annotations = prompt_annotations[['image_id', 'segmentation', 'category_id']]
        prompt_mask = utils.get_prompt_mask(mask_annotations, target.shape, self.resize, classes)

        # gt
        target_annotations = self.annotations[self.annotations.image_id == image_id['id']][['category_id', 'segmentation']]
        gt = self.resize(utils.get_gt(target_annotations, target.shape, classes).unsqueeze(0)).squeeze(0)

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
                c: ix for ix, c in enumerate(classes, start=1)
            }
        }

    def __len__(self):
        return len(self.images)


# main for testing the class
if __name__ == '__main__':
    from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

    preprocess = Compose([
        ToTensor(),
        #Resize((1000, 1000)),
    ])

    dataset = LabelAnythingDataset(
        instances_path='lvis_v1_train.json',
        preprocess=preprocess,
        num_max_examples=10,
        j_index_value=.1,
    )

    out = dataset[0]

    for k, v in out.items():
        print(f'{k}: {v.size()}' if isinstance(v, torch.Tensor) else f'{k}: {v}')