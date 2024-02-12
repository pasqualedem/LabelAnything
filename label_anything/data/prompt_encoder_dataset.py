from label_anything.data.coco import CocoLVISDataset
from torchvision.transforms import ToTensor
import torch
from typing import Optional
from label_anything.data.utils import PromptType, BatchKeys, flags_merge
import label_anything.data.utils as data_utils
from safetensors.torch import load_file
import random


class PromptEncoderDataset(CocoLVISDataset):
    def __init__(
            self,
            name,
            instances_path,
            clip_emb_dir,
            num_examples,
            img_dir=None,
            emb_dir: Optional[str] = None,
            max_points_per_annotation: int = 10,
            max_points_annotations: int = 50,
            preprocess=ToTensor(),
            load_gts: bool = False,
            do_subsample: bool = True,
            add_box_noise: bool = True,
            dtype=torch.float32,
    ):
        super().__init__(
            name=name,
            instances_path=instances_path,
            img_dir=img_dir,
            emb_dir=emb_dir,
            max_points_per_annotation=max_points_per_annotation,
            max_points_annotations=max_points_annotations,
            preprocess=preprocess,
            load_gts=load_gts,
            do_subsample=do_subsample,
            add_box_noise=add_box_noise,
            dtype=dtype,
        )
        self.clip_emb_dir = clip_emb_dir
        self.n_images = num_examples

    def _load_clip_embeddings(self, img_id):
        f = load_file(f"{self.clip_emb_dir}/{str(img_id).zfill(12)}.safetensors")
        return f['clip_embedding']

    def __getitem__(self, class_idx) -> dict:
        # extract randon images for class class_id
        class_idx = list(self.categories.keys())[class_idx]
        cat_id = self.categories[class_idx].get('id')
        img_ids = random.choices(population=list(self.cat2img[cat_id]), k=self.n_images)

        # get base image data
        images, image_key, ground_truths = self._get_images_or_embeddings(img_ids)

        # load image prompts
        bboxes, masks, points, classes, img_sizes = self._get_prompts(
            img_ids, [cat_id], possible_prompt_types=[PromptType.MASK, PromptType.BBOX, PromptType.POINT]
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

        flag_examples = flags_merge(flag_masks, flag_points, flag_bboxes)

        # load clip embeddings
        clip_embeddings = torch.stack([self._load_clip_embeddings(img_id) for img_id in img_ids])
        return {
            image_key: images,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.PROMPT_POINTS: points,
            BatchKeys.FLAG_POINTS: flag_points,
            BatchKeys.PROMPT_BBOXES: bboxes,
            BatchKeys.FLAG_BBOXES: flag_bboxes,
            BatchKeys.CLIP_EMBEDDINGS: clip_embeddings
        }

    def __len__(self):
        return len(self.categories)


def collate_fn(batched_input: list[dict[BatchKeys, torch.Tensor]]) -> dict[BatchKeys, torch.Tensor]:
    # collate images or embeddings
    image_key = BatchKeys.IMAGES if BatchKeys.IMAGES in batched_input[0].keys() else BatchKeys.EMBEDDINGS
    images = torch.cat([x[image_key] for x in batched_input], dim=0)

    # collate flag examples
    flag_examples = [x[BatchKeys.FLAG_EXAMPLES] for x in batched_input]
    flag_examples = data_utils.collate_flag_examples(flag_examples, len(flag_examples))

    # collate masks
    masks = [x[BatchKeys.PROMPT_MASKS] for x in batched_input]
    flags = [x[BatchKeys.FLAG_MASKS] for x in batched_input]

    masks, flag_masks = data_utils.collate_class_masks(masks, flags, len(masks))

    # collate bbox
    bboxes = [x[BatchKeys.PROMPT_BBOXES] for x in batched_input]
    flags = [x[BatchKeys.FLAG_BBOXES] for x in batched_input]
    max_annotations = max(x.size(2) for x in bboxes)
    bboxes, flag_bboxes = data_utils.collate_class_bbox(bboxes, flags, len(bboxes), max_annotations)

    # collate coords
    points = [x[BatchKeys.PROMPT_POINTS] for x in batched_input]
    flags = [x[BatchKeys.FLAG_POINTS] for x in batched_input]
    max_annotations = max(x.size(2) for x in points)

    points, flag_points = data_utils.collate_class_points(points, flags, len(points), max_annotations)

    # collate clip embeddings
    clip_embeddings = torch.stack([x[BatchKeys.CLIP_EMBEDDINGS] for x in batched_input])

    return {
        image_key: images.unsqueeze(dim=0),
        BatchKeys.FLAG_EXAMPLES: flag_examples,
        BatchKeys.PROMPT_MASKS: masks,
        BatchKeys.FLAG_MASKS: flag_masks,
        BatchKeys.PROMPT_BBOXES: bboxes,
        BatchKeys.FLAG_BBOXES: flag_bboxes,
        BatchKeys.PROMPT_POINTS: points,
        BatchKeys.FLAG_POINTS: flag_points,
        BatchKeys.CLIP_EMBEDDINGS: clip_embeddings,
    }
