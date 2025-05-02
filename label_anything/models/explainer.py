import math
from typing import Dict, List, Tuple
from einops import rearrange
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients, Saliency, LayerGradCam

from label_anything.data.utils import BatchKeys
from label_anything.demo.utils import debug_write
from label_anything.models.lam import Lam
from label_anything.models.mask_decoder import MaskDecoderLam


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
    BatchKeys.FLAG_EXAMPLES,
    BatchKeys.DIMS,
]

class LamLayerGradCam(LayerGradCam):
    def attribute(self, *args, **kwargs):
        inputs = args[0]
        b, m, c, h, w = inputs[0].shape
        attrs = super().attribute(*args, **kwargs)
        debug_write("attrs", attrs)
        size = int(math.sqrt(attrs[0].shape[0]))
        return (rearrange(attrs.sum(dim=-1), " (b m c)(h w) -> b m c h w", h=size, b=b, m=m, c=c),)

class LamExplainer(nn.Module):
    methods = {
        "integrated gradients": (IntegratedGradients, {}, {"n_steps": 100, "internal_batch_size": 1}),
        "saliency": (Saliency, {}, {}),
        "gradcam": (LamLayerGradCam, {}, {"attr_dim_summation": False}),
    }
    def __init__(self, model: Lam, method: str = "ig"):
        super(LamExplainer, self).__init__()
        self.model = model
        self.fc = nn.Linear(1, 1, bias=False)
        self.query_embeddings = None
        
        method, init_kwargs, kwargs = self.methods[method]
        if method == LamLayerGradCam:
            init_kwargs["layer"] = self.model.prompt_encoder.transformer.layers[-1].cross_attn_image_to_token.q_proj
        self.method = method(self, **init_kwargs)
        self.method_kwargs = kwargs

    def prepare(self, query_img, coord):
        x, y = coord
        feat_map = self.model.neck(self.model.image_encoder(query_img))
        x_multiplier = feat_map.shape[-1] / query_img.shape[-1]
        y_multiplier = feat_map.shape[-2] / query_img.shape[-2]
        x = int(x * x_multiplier)
        y = int(y * y_multiplier)
        self.query_embeddings = feat_map
        # Needed due to some trasformation applied to query embeddings
        upscaled_feat_map = self.model.mask_decoder.output_upscaling(feat_map)
        query_embeddings = self.model.mask_decoder.spatial_convs(upscaled_feat_map)

        # Step2
        feature_vector = query_embeddings[0, :, x, y]
        self.fc.weight = nn.Parameter(feature_vector)

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
        
        b, c, h, w = self.query_embeddings.shape
        mask_decoder: MaskDecoderLam = self.model.mask_decoder
        class_embeddings = mask_decoder._get_pe_result(pe_result)
        class_embeddings, self.query_embeddings = mask_decoder.transformer(
            self.query_embeddings, self.model.get_dense_pe(), class_embeddings
        )
        self.query_embeddings = rearrange(self.query_embeddings, "b (h w) c -> b c h w", h=h, w=w)

        class_embeddings = mask_decoder.class_mlp(class_embeddings)
        return self.fc(class_embeddings)

    def explain(self, batch, coord, target):
        batch = batch.copy()
        query_img = batch[BatchKeys.IMAGES][:, 0]
        support_imgs = batch[BatchKeys.IMAGES][:, 1:]
        batch[BatchKeys.IMAGES] = support_imgs
        self.prepare(query_img, coord)
        # Get a tuple from the batch
        main_input = {key: batch[key] for key in batch.keys() if key in GRADIENT_KEYS}

        # Remove flag 0 tensors
        if BatchKeys.FLAG_MASKS in batch and batch[BatchKeys.FLAG_MASKS].sum() == 0:
            del main_input[BatchKeys.PROMPT_MASKS]
        if BatchKeys.FLAG_BBOXES in batch and batch[BatchKeys.FLAG_BBOXES].sum() == 0:
            del main_input[BatchKeys.PROMPT_BBOXES]
        if BatchKeys.FLAG_POINTS in batch and batch[BatchKeys.FLAG_POINTS].sum() == 0:
            del main_input[BatchKeys.PROMPT_POINTS]

        additional_input = {
            key: batch[key] for key in batch.keys() if key in NON_GRADIENT_KEYS
        }
        tuple_mapping = {key: i for i, key in enumerate(main_input.keys())} | {
            key: i + len(main_input) for i, key in enumerate(additional_input.keys())
        }
        main_input = tuple(main_input.values())
        additional_input = tuple(additional_input.values())
        additional_input += (tuple_mapping,)

        attribution_tuple = self.method.attribute(
            main_input,
            additional_forward_args=additional_input,
            target=target,
            **self.method_kwargs,
        )
        return dict(zip(tuple_mapping.keys(), attribution_tuple))
