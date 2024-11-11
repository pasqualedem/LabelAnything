import os
from pathlib import Path

import torch
import torch.nn.functional as Ft
from einops import rearrange
from torchvision.transforms import functional as F

from label_anything.data.utils import BatchKeys
from label_anything.utils.utils import ResultDict

from .cfg import cfg
from .FewShotSegPartResnetSem import FewShotSegPart
from .SemiFewShotPartGraph import SemiFewShotSegPartGraph
from ..panet import unique_elements


class PPNet(FewShotSegPart):
    def postprocess_masks(self, logits, dims):
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        dims = dims[:, 0, :]  # get real sizes of the query images
        logits = [
            Ft.interpolate(
                torch.unsqueeze(logit, 0),
                size=dim.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for logit, dim in zip(logits, dims)
        ]

        logits = torch.cat(
            [
                Ft.pad(
                    mask,
                    (
                        0,
                        max_dims[1] - dims[i, 1],
                        0,
                        max_dims[0] - dims[i, 0],
                    ),
                    mode="constant",
                    value=float("-inf"),
                )
                for i, mask in enumerate(logits)
            ]
        )
        return logits

    def __init__(self, cfg, fold=0):
        cfg["exp_str"] += str(fold)
        cfg["ckpt_dir"] += str(fold)

        super(PPNet, self).__init__(cfg=cfg)

    def forward(self, batch):
        qry_imgs = [batch[BatchKeys.IMAGES][:, 0]]
        supp_imgs_tensor = batch[BatchKeys.IMAGES][:, 1:]
        b, m, c, _, _ = batch[BatchKeys.PROMPT_MASKS].shape
        c_fg = c - 1
        k = m // c_fg
        supp_imgs_tensor = rearrange(
            supp_imgs_tensor, "1 (k c) rgb h w -> c k rgb h w", k=k
        )

        supp_imgs = []
        for way in supp_imgs_tensor:
            way_imgs = [shot.unsqueeze(0) for shot in way]
            supp_imgs.append(way_imgs)
        masks = batch[BatchKeys.PROMPT_MASKS]  # B x M x C x H x W

        masks = masks.argmax(dim=2)
        masks = F.resize(
            masks,
            size=qry_imgs[0].shape[-2:],
            interpolation=F.InterpolationMode.NEAREST,
        )
        masks = rearrange(masks, "1 (k c) ... -> c k ...", k=k)
        classes = masks.shape[0]

        assert b == 1, "This implementation of PANet only supports batch size of 1"

        fore_masks = []
        back_masks = []
        class_ids = unique_elements([set(mask.unique()[1:].tolist()) for mask in masks])

        done_classes = []
        for class_id, mask_way in zip(class_ids, masks):
            done_classes.append(class_id)
            fore_masks_class = []
            back_masks_class = []
            # Create the whole mask once for all classes
            for mask_shot in mask_way:
                # Foreground mask: where mask equals class_id
                fore_mask = mask_shot.eq(class_id).float().unsqueeze(0)

                # Background mask: where mask does not equal class_id, excluding other class_ids
                back_mask = mask_shot.ne(class_id).float().unsqueeze(0)
                for j in range(1, classes + 1):
                    if j == class_id:
                        continue
                    back_mask = back_mask * (mask_shot.ne(j).float())

                fore_masks_class.append(fore_mask)
                back_masks_class.append(back_mask)

            fore_masks.append(fore_masks_class)
            back_masks.append(back_masks_class)
        assert len(set(done_classes)) == classes, "All classes should be processed"

        # reuse the supp_imgs as the unlabelled images
        # un_imgs = [[supp_img.clone() for supp_img in way] for way in supp_imgs]

        logits, logits_semantic, loss = super().forward(
            supp_imgs, fore_masks, back_masks, qry_imgs
        )  # , un_imgs)
        logits = self.postprocess_masks(logits, batch[BatchKeys.DIMS])
        return {ResultDict.LOGITS: logits}

    def predict(self, batch: dict):
        return self.forward(batch)

    def generate_class_embeddings(self, batch: dict, chunk_size=None):
        raise NotImplementedError("PANet does not support generating class embeddings")


def build_ppnet(ckpt_dir, fold=0, custom_preprocess=False):
    if custom_preprocess:
        raise NotImplementedError("Custom preprocess is not supported for PANet")
    model = PPNet(fold)
    cfg["ckpt_dir"] = ckpt_dir + str(fold)
    cfg["exp_str"] = cfg["exp_str"] + str(fold)
    cfg["resnet_init_path"] = Path(cfg["ckpt_dir"]).parent / "resnet"

    ckpt = os.path.join(f'{cfg["ckpt_dir"]}/best.pth')
    loaded_ckpt = torch.load(ckpt, map_location="cpu")
    # remove the "module." prefix from the keys
    new_state_dict = {}
    for k, v in loaded_ckpt.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model
