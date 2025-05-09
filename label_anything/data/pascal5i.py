from label_anything.data.utils import StrEnum, flags_merge
import torch
import label_anything.data.utils as utils
from label_anything.data.utils import BatchKeys, PromptType
from label_anything.data.pascal import PascalDataset
import random

from label_anything.data.examples import build_example_generator

class Pascal5iSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"

class Pascal5iDataset(PascalDataset):
    def __init__(
        self,
        val_fold_idx: int,
        n_folds: int,
        n_shots: int = None,
        val_num_samples: int = 1000,
        *args,
        **kwargs
    ):
        """Pascal 5i dataset.

        Args:
            val_fold_idx (int): Validation fold index.
            n_folds (int): Number of folds.
            n_shots (int): Number of shots.
        """
        super().__init__(*args, **kwargs, load_annotation_dicts=False)

        assert self.split in [Pascal5iSplit.TRAIN, Pascal5iSplit.VAL]
        assert val_fold_idx < n_folds
        assert self.split == Pascal5iSplit.TRAIN or n_shots is not None
        # If n_shots is min, n_ways should be max
        assert (
            n_shots != "min" or self.n_ways == "max"
        ), "If n_shots is min, n_ways should be max"
        assert self.n_ways != "max" or (
            n_shots == "min" or n_shots is None
        ), "If n_ways is max, n_shots should be min"

        self.val_fold_idx = val_fold_idx
        self.n_folds = n_folds
        self.n_shots = n_shots
        self.val_num_samples = val_num_samples
        self.__prepare_benchmark()

    def __prepare_benchmark(self):
        n_categories = len(self.categories)
        idxs_val = [
            self.val_fold_idx * (n_categories // self.n_folds) + i
            for i in range((n_categories // self.n_folds))
        ]
        idxs_train = [i for i in range(n_categories) if i not in idxs_val]

        if self.split == Pascal5iSplit.TRAIN:
            idxs = idxs_train
        else:
            idxs = idxs_val
        
        self.categories = {
            k: v for i, (k, v) in enumerate(self.categories.items()) if i in idxs
        }

        (
            self.img2cat,
            self.cat2img,
        ) = self._load_annotation_dicts()
        self.image_names = list(self.img2cat.keys())

        self.example_generator = build_example_generator(
            n_ways=self.n_ways,
            n_shots=self.n_shots,
            categories_to_imgs=self.cat2img,
            images_to_categories=self.img2cat,
        )

    def __getitem__(self, idx_batchmetadata: tuple[int, int]) -> dict:
        if self.split == Pascal5iSplit.TRAIN or self.n_shots is "min":
            return super().__getitem__(idx_batchmetadata)
        elif self.split == Pascal5iSplit.VAL:
            idx, metadata = idx_batchmetadata
            intended_classes = [[] for _ in range(self.n_ways * self.n_shots + 1)]
            if self.n_ways == 1:
                cat_ids = [-1, random.choice(list(self.categories.keys()))]
                images_data = random.sample(
                    list(self.cat2img[cat_ids[1]]), self.n_shots + 1
                )
                intended_classes[0].append(cat_ids[1])
                for i in range(1, self.n_shots + 1):
                    intended_classes[i].append(cat_ids[1])
            else:
                cat_ids = random.sample(list(self.categories.keys()), self.n_ways)
                query_image_name = random.choice(list(self.cat2img[cat_ids[0]]))
                intended_classes[0].append(cat_ids[0])
                
                images_data = [query_image_name]
                for cat_id in cat_ids:
                    cat_image_names = list(self.cat2img[cat_id])
                    cat_image_names = random.sample(cat_image_names, self.n_shots)
                    for i in (range(len(images_data), len(images_data) + self.n_shots)):
                        intended_classes[i].append(cat_id)
                    images_data += cat_image_names
                cat_ids = [-1] + sorted(cat_ids)
                
            images_name, _ = zip(*images_data)

            images, image_key, ground_truths = self._get_images_or_embeddings(images_name)

            masks, classes, img_sizes = self._get_prompts(images_data, cat_ids, with_random_choice=False)

        masks, flag_masks = utils.annotations_to_tensor(
            self.prompts_processor, masks, img_sizes, PromptType.MASK
        )

        if ground_truths is None:
            ground_truths = self.compute_ground_truths(images_data, img_sizes, cat_ids, with_random_choice=False)

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
            (len(images_data), len(cat_ids), 1, 4), dtype=torch.float32
        )
        flag_bboxes = torch.zeros(
            (len(images_data), len(cat_ids), 1), dtype=torch.uint8
        )
        prompt_points = torch.zeros(
            (len(images_data), len(cat_ids), 1, 2), dtype=torch.float32
        )
        flag_points = torch.zeros(
            (len(images_data), len(cat_ids), 1), dtype=torch.uint8
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
            BatchKeys.INTENDED_CLASSES: intended_classes,
            BatchKeys.IMAGE_IDS: images_data,
            BatchKeys.GROUND_TRUTHS: ground_truths,
        }
        return data_dict
    
    def __len__(self):
        if self.split == Pascal5iSplit.TRAIN:
            return super().__len__()
        elif self.split == Pascal5iSplit.VAL:
            return self.val_num_samples