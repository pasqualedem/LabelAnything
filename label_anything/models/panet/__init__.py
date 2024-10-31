from einops import rearrange, repeat
import torch
import torch.nn.functional as Ft
from torchvision.transforms import functional as F

from label_anything.data.utils import BatchKeys
from label_anything.utils.utils import ResultDict
from .fewshot import FewShotSeg


class PANet(FewShotSeg):
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

    
    def __init__(self, in_channels=3, pretrained_path=None, pretrained_encoder_path=None, cfg=None):
        if cfg is None:
            cfg = {'align': False}
        super().__init__(in_channels, pretrained_path=pretrained_encoder_path, cfg=cfg)
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            try:
                self.load_state_dict(ckpt)
                print(f"Loaded model from {pretrained_path}")
            except RuntimeError:
                # Remove the "module." prefix
                ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
                self.load_state_dict(ckpt)
            print(f"Loaded model from {pretrained_path}")
        
    def forward(self, batch: dict):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        qry_imgs = [batch[BatchKeys.IMAGES][:, 0]]
        supp_imgs_tensor = batch[BatchKeys.IMAGES][:, 1:]
        b, m, c, _, _ = batch[BatchKeys.PROMPT_MASKS].shape
        c_fg = c - 1
        k = m // c_fg
        supp_imgs_tensor = rearrange(supp_imgs_tensor, '1 (k c) rgb h w -> c k rgb h w', k=k)
        
        start.record()
        supp_imgs = []
        for way in supp_imgs_tensor:
            way_imgs = [shot.unsqueeze(0) for shot in way]
            supp_imgs.append(way_imgs)
        masks = batch[BatchKeys.PROMPT_MASKS]  # B x M x C x H x W
        end.record()
        torch.cuda.synchronize()
        elapsed_phase_one = start.elapsed_time(end)

        start.record()

        masks = masks.argmax(dim=2)
        masks = F.resize(masks, size=qry_imgs[0].shape[-2:], interpolation=F.InterpolationMode.NEAREST)
        masks = rearrange(masks, "1 ... -> ...")
        classes = masks.shape[1]

        assert b == 1, "This implementation of PANet only supports batch size of 1"

        fore_masks = []
        back_masks = []
        for mask in masks:
            fore_masks_class = []
            back_masks_class = []
            for class_id in range(classes):
                fore_mask = torch.where(mask == class_id, torch.ones_like(mask), torch.zeros_like(mask))
                back_mask = torch.where(mask != class_id, torch.ones_like(mask), torch.zeros_like(mask))
                for class_id_back in range(classes):
                    back_mask[mask == class_id_back] = 0
                fore_masks_class.append(fore_mask.unsqueeze(0))
                back_masks_class.append(back_mask.unsqueeze(0))
            fore_masks.append(fore_masks_class)
            back_masks.append(back_masks_class) 
        end.record()
        torch.cuda.synchronize()
        elapsed_phase_two = start.elapsed_time(end)

        start.record()
        logits, loss = super().forward(supp_imgs, fore_masks, back_masks, qry_imgs)
        logits = self.postprocess_masks(logits, batch[BatchKeys.DIMS])
        end.record()
        torch.cuda.synchronize()
        elapsed_phase_three = start.elapsed_time(end)
        print(f"Elapsed time phase one: {elapsed_phase_one}")
        print(f"Elapsed time phase two: {elapsed_phase_two}")
        print(f"Elapsed time phase three: {elapsed_phase_three}")
        return {ResultDict.LOGITS: logits}
        
    
    def predict(self, batch: dict):
        return self.forward(batch)
    
    def generate_class_embeddings(self, batch: dict, chunk_size=None):
        raise NotImplementedError("PANet does not support generating class embeddings")
    
def build_panet(cfg=None, pretrained_path=None, custom_preprocess=False):
    if custom_preprocess:
        raise NotImplementedError("Custom preprocess is not supported for PANet")
    return PANet(cfg=cfg, pretrained_path=pretrained_path)