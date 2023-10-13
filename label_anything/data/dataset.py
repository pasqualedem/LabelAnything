import torchvision.transforms
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import random
import torch
from torchvision.transforms import Resize, ToTensor, InterpolationMode, Compose
import warnings
import utils
from typing import Union, Dict, List, Tuple, Any

warnings.filterwarnings('ignore')


class LabelAnythingDataset(Dataset):
    def __init__(
            self,
            instances_path: str,  # Path
            directory: Union[str, None] = None,  # directory (only if images have to be loaded from disk)
            num_max_examples: int = 10,  # number of max examples to be given for the target image
            preprocess: Union[Compose, ToTensor] = ToTensor(),  # preprocess step
            j_index_value: float = .5,  # threshold for extracting examples
            seed: int = 42,  # for reproducibility
            resize_dim: int = 224,  # shape for stacking images
            max_mum_coords: int = 10,  # max number of coords for each example for each class
    ) -> None:
        """
        LabelAnythingDataset is responsible for data fetching process.

        Arguments:
             instances_path: path of entire data set, including images, annotations, and categories.
             directory: path to image directory.
             num_max_examples: maximum number of examples that are generated for a single query image.
             preprocess: preprocessing phase.
             j_index_value: threshold value for discarding low related examples.
             seed: int value for reproducibility.
             resize_dim: int value for final dimension of images.
             max_mum_coords: maximum number of coordinates to extract at each batch.
        """
        super().__init__()
        instances = utils.load_instances(instances_path)
        self.annotations = pd.DataFrame(instances['annotations'])
        self.images = pd.DataFrame(instances['images'])
        self.load_from_dir = directory is not None
        self.categories = pd.DataFrame(instances['categories'])
        self.num_max_examples = num_max_examples
        assert self.num_max_examples > 1, "maximum number of coords must be greater then 1."
        self.directory = directory
        self.preprocess = preprocess
        self.j_index_value = j_index_value
        self.seed = seed
        self.resize = Resize((resize_dim, resize_dim))
        self.resize_mask = Resize((resize_dim, resize_dim), interpolation=InterpolationMode.NEAREST)
        self.img_shape = (1, resize_dim, resize_dim)
        self.max_num_coords = max_mum_coords
        assert self.max_num_coords > 1, "maximum number of coords must be greater then 1."
        self.num_coords = random.randint(1, max_mum_coords)
        self.num_examples = random.randint(1, self.num_max_examples)

    def __load_image(
            self,
            img: pd.Series
    ) -> Image:
        """
        Loads an image from disk or downloading from the web, using the cooc url.
        """
        if self.load_from_dir:
            return Image.open(f'{self.directory}/{img["id"]}.jpg')
        return Image.open(BytesIO(requests.get(img["coco_url"]).content))

    def __extract_examples(
            self,
            img_id: int
    ) -> Tuple[List[int], List[int]]:
        """
        Extracts similar examples to query image, represented by image id, filtering out those images that have few
        classes in common with the query image.

        Arguments:
             img_id: id of query image.

         Returns:
             List[int]: image ids of similar image examples.
             List[int]: unique target class ids.
        """
        target_classes = self.annotations[self.annotations.image_id == img_id]['category_id'].unique().tolist()
        class_projection = self.annotations[['image_id', 'category_id']].groupby(by='image_id')['category_id'].apply(
            list).reset_index(name='category_id')
        class_projection['j_score'] = class_projection.apply(
            lambda x: utils.compute_j_index(target_classes, x.category_id),
            axis=1)
        class_projection = class_projection[class_projection['j_score'] > self.j_index_value]
        return class_projection.sample(n=self.num_examples, replace=True,
                                       random_state=self.seed).image_id.tolist(), list(target_classes)

    def __getitem__(
            self,
            item: int
    ) -> Dict[str, Any]:
        """
        Override get item method.

        Arguments:
            item: i-th query image to fetch.

        Returns:
            Dict[str, Any]: a dictionary with the following keys.
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
                'flag_bbox'_ example image bounding box flags as a torch tensor of shape M x C x N x 4, where M is the
                    number of examples extracted for the given query image, C is the number of classes associated to the
                    given image, and N is the maximum number of annotations associated to a pair (image, class).
                'gt': query image classes mask as a tensor of shape H x W, in which each pixel has a certain value k if
                    that pixel is in the mask of the k-th class associated to the query image.
                'classes': dict in which each pair k: v is ith class corresponding to class id.
        """
        image_id = self.images.iloc[item]  # image row id
        target = self.__load_image(image_id)  # load image
        target = target if not self.preprocess else self.preprocess(target)  # preprocess

        # choose similar content with respect to target image
        example_ids, classes = self.__extract_examples(image_id['id'])
        examples = [self.__load_image(row)
                    for ix, row in self.images[self.images.id.isin(example_ids)].iterrows()]  # load images
        if self.preprocess:
            examples = [self.preprocess(x) for x in examples]

        # annotations useful for prompt
        prompt_annotations = self.annotations[(self.annotations.image_id.isin(example_ids)) &
                                              (self.annotations.category_id.isin(classes))]

        # bboxes
        bbox_annotations = prompt_annotations[['image_id', 'bbox', 'category_id']]
        prompt_bbox, flag_bbox = utils.get_prompt_bbox(bbox_annotations, classes)

        # masks
        mask_annotations = prompt_annotations[['image_id', 'segmentation', 'category_id']]
        example_shape = [x.shape for x in examples]
        prompt_mask = utils.get_prompt_mask(mask_annotations, example_shape, self.resize_mask, classes).squeeze(dim=2)

        # gt
        target_annotations = self.annotations[self.annotations.image_id == image_id['id']][
            ['category_id', 'segmentation']]
        gt = self.resize_mask(utils.get_gt(target_annotations, target.shape, classes).unsqueeze(0)).squeeze(0)

        examples = [self.resize(x) for x in examples]  # resize images
        examples = torch.stack(examples)  # stack images

        prompt_coords, flag_coords = utils.get_prompt_coords(annotations=mask_annotations,
                                                             target_classes=classes,
                                                             num_coords=self.num_coords,
                                                             original_shape=example_shape,
                                                             resize=self.resize_mask)

        target = self.resize(target)

        return {
            'target': target,  # 3 x h x w
            'examples': examples,  # m x 3 x h x w
            'prompt_mask': prompt_mask,  # m x c x h x w
            'prompt_coords': prompt_coords,  # m x c x n x k x 2
            'flag_coords': flag_coords,  # m x c x n x k
            'prompt_bbox': prompt_bbox,  # m x c x n x 4
            'flag_bbox': flag_bbox,  # m x c x n
            'gt': gt,  # h x w
            'classes': {
                ix: c for ix, c in enumerate(classes, start=1)
            }
        }

    def __len__(self):
        """
        Returns the length od the dataset.
        """
        return len(self.images)

    def reset_num_coords(self):
        """
        Regenerates the number of coordinates to extract for each annotation selected.
        """
        self.num_coords = random.randint(1, self.max_num_coords)

    def reset_num_examples(self):
        """
        Regenerates the number of examples to extract for each query image selected.
        """
        self.num_examples = random.randint(1, self.num_max_examples)

    def collate_fn(
            self,
            batched_input: List[Dict[str, Any]]
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
        classes = [x['classes'] for x in batched_input]
        new_classes = utils.rearrange_classes(classes)

        # gt
        gts = [x['gt'] for x in batched_input]
        gts = torch.stack([utils.collate_gt(x, classes[ix], new_classes) for ix, x in enumerate(gts)])

        # prompt mask
        masks = [x['prompt_mask'] for x in batched_input]
        masks = torch.stack([utils.collate_mask(mask, classes[i], new_classes) for i, mask in enumerate(masks)])

        # prompt bbox
        bboxes = [x["prompt_bbox"] for x in batched_input]
        flags = [x["flag_bbox"] for x in batched_input]
        max_annotations = max(x.size(2) for x in bboxes)
        bboxes_flags = [utils.collate_bbox(bboxes[i], flags[i], classes[i], new_classes, max_annotations)
                        for i in range(len(bboxes))]
        bboxes = torch.stack([x[0] for x in bboxes_flags])
        bbox_flags = torch.stack([x[1] for x in bboxes_flags])

        # prompt coords
        coords = [x['prompt_coords'] for x in batched_input]
        flags = [x['flag_coords'] for x in batched_input]
        coords_flags = [utils.collate_coords(coords[i], flags[i], classes[i], new_classes, max_annotations)
                        for i in range(len(coords))]
        coords = torch.stack([x[0] for x in coords_flags])
        coord_flags = torch.stack([x[1] for x in coords_flags])

        # query image
        query_image = torch.stack([x["target"] for x in batched_input])

        # example image
        example_images = torch.stack([x["examples"] for x in batched_input])

        data_dict = {
            'query_image': query_image,
            'example_images': example_images,
            'point_coords': coords,
            'point_flags': coord_flags,
            'boxes': bboxes,
            'box_flags': bbox_flags,
            'mask_inputs': masks
        }

        # reset dataset parameters
        self.reset_num_coords()
        self.reset_num_examples()

        return data_dict, gts


if __name__ == '__main__':
    from torchvision.transforms import Compose, ToTensor, Resize
    from torch.utils.data import DataLoader

    preprocess = Compose([
        ToTensor(),
    ])

    dataset = LabelAnythingDataset(
        instances_path='lvis_v1_train.json',
        preprocess=preprocess,
        num_max_examples=10,
        j_index_value=.1,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    data_dict, gt = next(iter(dataloader))

    print([f'{k}: {v.size()}' for k, v in data_dict.items()])
    print(f'gt: {gt.size()}')
