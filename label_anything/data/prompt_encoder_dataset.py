from torch.utils.data import Dataset
from label_anything.data.coco import CocoLVISDataset
from torchvision.transforms import ToTensor
import torch


class PromptEncoderDataset(CocoLVISDataset):
    def __init__(
            self,
            name,
            instance_path,
            clip_emb_dir,
            img_dir=None,
            max_num_examples=10,
            max_points_annotations=50,
            preprocess=ToTensor(),
            seed=42,
            emb_dir=None,
            n_folds=-1,
            val_fold=-1,
            load_embeddings=False,
            load_gts=False,
            split="train",
            do_subsample=True,
            add_box_noise=True,
    ):
        super().__init__(
            name=name,
            instances_path=instance_path,
            img_dir=img_dir,
            max_num_examples=max_num_examples,
            max_points_annotations=max_points_annotations,
            preprocess=preprocess,
            seed=seed,
            emb_dir=emb_dir,
            n_folds=n_folds,
            val_fold=val_fold,
            load_embeddings=load_embeddings,
            load_gts=load_gts,
            split=split,
            do_subsample=do_subsample,
            add_box_noise=add_box_noise,
        )
        self.clip_emb_dir = clip_emb_dir

    def __getitem__(self, class_id) -> dict:
        # extract randon images for class class_id
        n_images = self.rng.randint(1, self.max_num_examples)
        img_ids = self.rng.choices(population=self.cat2img[class_id], k=n_images)

        # get base image data
        images, image_key, ground_truths = self._get_images_or_embeddings(img_ids)

        # load image prompts


    def __len__(self):
        return len(self.categories)
