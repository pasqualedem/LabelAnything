from label_anything.data.coco import CocoLVISDataset
from torchvision.transforms import ToTensor
import torch
from typing import Optional
from label_anything.data.utils import PromptType, BatchKeys
from safetensors.torch import load_file


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
            seed: int = 42,
            load_gts: bool = False,
            do_subsample: bool = True,
            add_box_noise: bool = True,
            prompt_types: list[PromptType] = [
                PromptType.BBOX,
                PromptType.MASK,
                PromptType.POINT,
            ],
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
            seed=seed,
            load_gts=load_gts,
            do_subsample=do_subsample,
            add_box_noise=add_box_noise,
            prompt_types=prompt_types,
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
        img_ids = self.rng.choices(population=list(self.cat2img[cat_id]), k=self.n_images)

        # get base image data
        images, image_key, ground_truths = self._get_images_or_embeddings(img_ids)

        # load image prompts
        bboxes, masks, points, classes, img_sizes = self._get_prompts(
            img_ids, [cat_id]
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
        dims = torch.tensor(img_sizes)

        # load clip embeddings
        clip_embeddings = torch.stack([self._load_clip_embeddings(img_id) for img_id in img_ids])
        return {
            image_key: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.PROMPT_POINTS: points,
            BatchKeys.FLAG_POINTS: flag_points,
            BatchKeys.PROMPT_BBOXES: bboxes,
            BatchKeys.FLAG_BBOXES: flag_bboxes,
            BatchKeys.DIMS: dims,
            BatchKeys.CLIP_EMBEDDINGS: clip_embeddings
        }

    def __len__(self):
        return len(self.categories)
