from einops import rearrange, repeat
import torch
import torch.nn.functional as Ft
from torchvision.transforms import functional as F

from label_anything.data.utils import BatchKeys
from label_anything.utils.utils import ResultDict
from .fewshot import FewShotSeg


def unique_elements(structure):
    # Flatten and get element frequency
    all_elements = [elem for s in structure for elem in s]
    element_count = {elem: all_elements.count(elem) for elem in all_elements}
    
    # Select unique element for each set based on count
    result = []
    for s in structure:
        if len(s) == 1:
            # Single element, take it directly
            result.append(next(iter(s)))
        else:
            # Take the element that appears least in all sets
            unique_elem = min(s, key=lambda x: element_count[x])
            # Update the count
            element_count[unique_elem] += 1
            result.append(unique_elem)
    
    return result

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
        qry_imgs = [batch[BatchKeys.IMAGES][:, 0]]
        supp_imgs_tensor = batch[BatchKeys.IMAGES][:, 1:]
        b, m, c, _, _ = batch[BatchKeys.PROMPT_MASKS].shape
        c_fg = c - 1
        k = m // c_fg
        supp_imgs_tensor = rearrange(supp_imgs_tensor, '1 (k c) rgb h w -> c k rgb h w', k=k)
        
        supp_imgs = []
        for way in supp_imgs_tensor:
            way_imgs = [shot.unsqueeze(0) for shot in way]
            supp_imgs.append(way_imgs)
        masks = batch[BatchKeys.PROMPT_MASKS]  # B x M x C x H x W

        masks = masks.argmax(dim=2)
        masks = F.resize(masks, size=qry_imgs[0].shape[-2:], interpolation=F.InterpolationMode.NEAREST)
        masks = rearrange(masks, "1 (k c) ... -> c k ...", k=k)
        classes = masks.shape[0]

        assert b == 1, "This implementation of PANet only supports batch size of 1"

        fore_masks = []
        back_masks = []
        unique_mask_elements = [set(mask.unique().tolist()) for mask in masks]
        # Remove background class
        unique_mask_elements = [set(mask_elements) - {0} for mask_elements in unique_mask_elements]
        class_ids = unique_elements(unique_mask_elements)

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
                for j in range(1, classes+1):
                    if j == class_id:
                        continue
                    back_mask = back_mask * (mask_shot.ne(j).float())

                fore_masks_class.append(fore_mask)
                back_masks_class.append(back_mask)

            fore_masks.append(fore_masks_class)
            back_masks.append(back_masks_class)
        assert len(set(done_classes)) == classes, "All classes should be processed"

        logits, loss = super().forward(supp_imgs, fore_masks, back_masks, qry_imgs)
        logits = self.postprocess_masks(logits, batch[BatchKeys.DIMS])
        return {ResultDict.LOGITS: logits}
        
    
    def predict(self, batch: dict):
        return self.forward(batch)
    
    def generate_class_embeddings(self, batch: dict, chunk_size=None):
        raise NotImplementedError("PANet does not support generating class embeddings")
    
def build_panet(cfg=None, pretrained_path=None, custom_preprocess=False):
    if custom_preprocess:
        raise NotImplementedError("Custom preprocess is not supported for PANet")
    return PANet(cfg=cfg, pretrained_path=pretrained_path)