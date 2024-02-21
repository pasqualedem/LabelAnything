import random

import torch

import label_anything.data.utils as utils
from label_anything.data.coco import CocoLVISDataset
from label_anything.data.examples import build_example_generator
from label_anything.data.utils import AnnFileKeys, BatchKeys, BatchMetadataKeys, PromptType, StrEnum, flags_merge


class Coco20iSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"


class Coco20iDataset(CocoLVISDataset):
    def __init__(
        self,
        split: Coco20iSplit,
        val_fold_idx: int,
        n_folds: int,
        n_shots: int = None,
        *args,
        **kwargs
    ):
        """COCO 20i dataset.

        Args:
            split (Coco20iSplit): Split to use (can be train or val).
            val_fold_idx (int): Validation fold index.
            n_folds (int): Number of folds.
            n_shots (int): Number of shots.
        """
        super().__init__(*args, **kwargs)

        assert split in [Coco20iSplit.TRAIN, Coco20iSplit.VAL]
        assert val_fold_idx < n_folds
        assert split == Coco20iSplit.TRAIN or n_shots is not None

        self.split = split
        self.val_fold_idx = val_fold_idx
        self.n_folds = n_folds
        self.n_shots = n_shots
        self._prepare_benchmark()

    def _prepare_benchmark(self):
        """Prepare the benchmark by selecting the categories to use and
        updating the annotation dicts.
        """
        n_categories = len(self.categories)
        idxs_val = [
            self.val_fold_idx + i * self.n_folds
            for i in range(n_categories // self.n_folds)
        ]
        idxs_train = [i for i in range(n_categories) if i not in idxs_val]

        # keep categories corresponding to the selected indices
        # self.categories is a dict
        if self.split == Coco20iSplit.TRAIN:
            idxs = idxs_train
        elif self.split == Coco20iSplit.VAL:
            idxs = idxs_val
        self.categories = {
            k: v for i, (k, v) in enumerate(self.categories.items()) if i in idxs
        }

        # update dicts
        (
            self.image_annotations,
            self.img2cat,
            self.img2cat_annotations,
            self.cat2img,
            self.cat2img_annotations,
        ) = self._load_annotation_dicts()

        # load image ids and info
        instances = utils.load_instances(self.instances_path)
        img2cat_keys = set(self.img2cat.keys())
        self.images = {
            x[AnnFileKeys.ID]: x
            for x in instances[AnnFileKeys.IMAGES]
            if x[AnnFileKeys.ID] in img2cat_keys
        }
        self.image_ids = list(self.images.keys())

        # example generator/selector
        self.example_generator = build_example_generator(
            n_ways=self.n_ways,
            n_shots=self.n_shots,
            categories_to_imgs=self.cat2img
        )

    def __getitem__(self, idx_batchmetadata: tuple[int, int]) -> dict:
        """Get an item from the dataset. Preserves the original functionality
        of the COCO dataset for the train split. For the val split, it samples
        a random category and returns the corresponding images (n_shots + 1).

        Args:
            idx_num_examples (tuple[int, int]): Index and number of examples to
                return.

        Returns:
            dict: Data dictionary.
        """
        if self.split == Coco20iSplit.TRAIN or self.n_shots == "min":
            return super().__getitem__(idx_batchmetadata)
        elif self.split == Coco20iSplit.VAL:
            idx, metadata = idx_batchmetadata
            # sample a random category
            cat_ids = [-1, random.choice(list(self.categories.keys()))]
            # sample random img ids
            image_ids = random.sample(list(self.cat2img[cat_ids[1]]), self.n_shots + 1)

            # load, stack and preprocess the images
            images, image_key, ground_truths = self._get_images_or_embeddings(image_ids)

            # create the prompt dicts
            bboxes, masks, points, classes, img_sizes = self._get_prompts(
                image_ids, cat_ids, metadata[BatchMetadataKeys.PROMPT_TYPES]
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
            
            flag_examples = flags_merge(flag_masks, flag_points, flag_bboxes)

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
                BatchKeys.FLAG_EXAMPLES: flag_examples,
                BatchKeys.DIMS: dims,
                BatchKeys.CLASSES: classes,
                BatchKeys.IMAGE_IDS: image_ids,
                BatchKeys.GROUND_TRUTHS: ground_truths,
            }

            return data_dict

    def __len__(self):
        if self.split == Coco20iSplit.TRAIN:
            return super().__len__()
        elif self.split == Coco20iSplit.VAL:
            return 1000
