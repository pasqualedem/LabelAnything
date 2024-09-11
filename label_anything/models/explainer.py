from typing import Dict, List, Tuple
from einops import rearrange
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

from label_anything.data.utils import BatchKeys
from label_anything.models.lam import Lam


GRADIENT_KEYS = [
    BatchKeys.IMAGES,
    BatchKeys.PROMPT_BBOXES,
    BatchKeys.PROMPT_MASKS,
    BatchKeys.PROMPT_POINTS,
]
NON_GRADIENT_KEYS = [
    BatchKeys.FLAG_POINTS,
    BatchKeys.FLAG_BBOXES,
    BatchKeys.FLAG_MASKS,
    BatchKeys.DIMS,
]

class LamExplainer(nn.Module):
    def __init__(self, model: Lam):
        super(LamExplainer, self).__init__()
        self.model = model
        self.fc = nn.Linear(512, 1)
        
    def prepare(self, query_img, coord):
        x, y = coord
        feat_map = self.model.neck(self.model.image_encoder(query_img))
        x_multiplier = feat_map.shape[-1] / query_img.shape[-1]
        y_multiplier = feat_map.shape[-2] / query_img.shape[-2]
        x = int(x * x_multiplier)
        y = int(y * y_multiplier)
        # Needed due to some trasformation applied to query embeddings
        upscaled_feat_map = self.model.mask_decoder.output_upscaling(feat_map)
        query_embeddings = self.model.mask_decoder.spatial_convs(upscaled_feat_map)

        self.model.mask_decoder.query_embeddings = feat_map
        #print('Feature map:', feat_map.shape)

        # Step2
        feature_vector = query_embeddings[0, :, x, y]
        self.fc.weight = nn.Parameter(feature_vector)
        
    def _classify(self, query_embeddings, class_embeddings, flag_examples):
        b, d, h, w = query_embeddings.shape
        _, c, _ = class_embeddings.shape
        seg = self.fc(class_embeddings)
        '''seg = (class_embeddings @ self.query_embeddings.view(b, d, h * w)).view(
            b, -1, h, w
        )'''
        if self.model.mask_decoder.segment_example_logits:
            seg = rearrange(seg, "b (n c) h w -> b n c h w", c=c)
            seg[flag_examples.logical_not()] = float("-inf")
            seg = seg.max(dim=1).values
        return seg

    def forward(
    self, *batched_input_tuple: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, torch.Tensor]]:
        
        tuple_mapping = batched_input_tuple[-1]

        # Prepare batched_input in the format expected by self._forward
        batched_input = {
            key: batched_input_tuple[value] for key, value in tuple_mapping.items()
        }

        prompt_embeddings = self.model.prepare_embeddings_example(batched_input)
        points, boxes, masks, flag_examples = self.model.prepare_prompts(batched_input)

        pe_result = self.model.prompt_encoder(
            image_embeddings=prompt_embeddings,
            points=points,
            boxes=boxes,
            masks=masks,
            flag_examples=flag_examples,
        )

        # seg = self.model.mask_decoder.forward_captum(
        #     support_embeddings=prompt_embeddings,
        #     image_pe=self.model.get_dense_pe(),
        #     class_embeddings=pe_result,
        #     flag_examples=flag_examples,
        # )
        class_embeddings = self.model._get_class_embeddings(pe_result)

        class_embeddings, self.query_embeddings = self.transformer(
            self.query_embeddings, self.model.get_dense_pe(), class_embeddings
        )
        self.query_embeddings = rearrange(self.query_embeddings, "b (h w) c -> b c h w", h=h)

        upscaled_embeddings, class_embeddings = self.model.mask_decoder._upscale(
            self.query_embeddings, class_embeddings
        )
        upscaled_embeddings = self.model.mask_decoder._spatial_convs(upscaled_embeddings)
        seg = self._classify(upscaled_embeddings, class_embeddings, flag_examples)

        if "flag_gts" in batched_input:
            seg[batched_input["flag_gts"].logical_not()] = -1 * torch.inf
    
        return seg
    
    def explain(self, batch, coord, target):
        query_img = batch[BatchKeys.IMAGES][0]
        self.prepare(query_img, coord)
        ig = IntegratedGradients(self)
        # Get a tuple from the batch
        tuple_mapping = {key: i for i, key in enumerate(batch.keys())}
        input = tuple(batch[key]for key in batch.keys() if key in GRADIENT_KEYS)
        additional_input = tuple(batch[key] for key in batch.keys() if key in NON_GRADIENT_KEYS)
        additional_input = tuple(additional_input + (tuple_mapping,))
        
        attr = ig.attribute(input, additional_forward_args=additional_input, target=target)
        return attr
    