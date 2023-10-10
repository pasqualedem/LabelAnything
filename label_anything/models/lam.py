# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptImageEncoder


class Lam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptImageEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        LAM predicts object masks from an image and a list of examples images with prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts with their corresponding images.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (dict): a dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'target_image': The target image as a torch tensor in Bx3xHxW format,
                already transformed for input to the model.
              'example_images': The example images as a torch tensor in BxNx3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxMxCxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape BxMxCx4.
                Already transformed to the input frame of the model.
              'box_flags': (torch.Tensor) Batched box flags, with shape BxMxC.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form BxMxCxHxW.

        Returns:
          torch.Tensor: Batched multiclass mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is the number of classes, (H, W) is the
                original size of the image.
        """
        B, N = batched_input['example_images'].shape[:2]
        prompt_images = rearrange(batched_input['example_images'], 'b n c h w -> (b n) c h w')
        images = torch.cat((batched_input['target_image'], prompt_images), dim=0)
        image_embeddings = self.image_encoder(images)
        prompt_images_embeddings = rearrange(image_embeddings[B:], '(b n) c h w -> b n c h w', b=B)
        image_embeddings = image_embeddings[:B]

        if "point_coords" in batched_input:
            points = (batched_input["point_coords"], batched_input["point_labels"])
        else:
            points = None
        if "boxes" in batched_input:
            boxes = batched_input["boxes"]
            box_flags = batched_input["box_flags"]
            boxes = (boxes, box_flags)
        else:
            boxes = None
        example_embeddings = self.prompt_encoder(
            image_embeddings=prompt_images_embeddings,
            points=points,
            boxes=boxes,
            masks=batched_input.get("mask_inputs", None),
        )
        
        seg = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            example_embeddings=example_embeddings
        )

        return seg

    def init_pretrained_weights(self, weights):
        """
        Initialize certain modules with pretrained weights from Sam.
        """
        # Load weights for the image encoder
        image_encoder_weights = {k[len("image_encoder."):]: v for k, v in weights.items() if k.startswith("image_encoder")}
        self.image_encoder.load_state_dict(image_encoder_weights)
        # Load weights for the prompt encoder
        pe_layer_weights = {k[len("prompt_encoder.pe_layer."):]: v for k, v in weights.items() if k.startswith("prompt_encoder.pe_layer")}
        self.prompt_encoder.pe_layer.load_state_dict(pe_layer_weights)
        point_embeddings_weights = {k[len("prompt_encoder.point_embeddings."):]: v for k, v in weights.items() if k.startswith("prompt_encoder.point_embeddings")}
        self.prompt_encoder.point_embeddings.load_state_dict(point_embeddings_weights)
        not_a_point_embed_weights = {k[len("prompt_encoder.not_a_point_embed."):]: v for k, v in weights.items() if k.startswith("prompt_encoder.not_a_point_embed")}
        self.prompt_encoder.not_a_point_embed.load_state_dict(not_a_point_embed_weights)
        mask_downscaling_weights = {k[len("prompt_encoder.mask_downscaling."):]: v for k, v in weights.items() if k.startswith("prompt_encoder.mask_downscaling")}
        self.prompt_encoder.mask_downscaling.load_state_dict(mask_downscaling_weights)
        no_mask_embed_weights = {k[len("prompt_encoder.no_mask_embed."):]: v for k, v in weights.items() if k.startswith("prompt_encoder.no_mask_embed")}
        self.prompt_encoder.no_mask_embed.load_state_dict(no_mask_embed_weights)

        # Load tranformer weights
        transformer_weights = {k[len("mask_decoder.transformer."):]: v for k, v in weights.items() if k.startswith("mask_decoder.transformer")}
        self.prompt_encoder.transformer.load_state_dict(transformer_weights)

        # Load weights for the mask decoder transformer
        self.mask_decoder.transformer.load_state_dict(transformer_weights.copy())

        # Load weights for the mask decoder output upscaling
        output_upscaling_weights = {k[len("mask_decoder.output_upscaling."):]: v for k, v in weights.items() if k.startswith("mask_decoder.output_upscaling")}
        self.mask_decoder.output_upscaling.load_state_dict(output_upscaling_weights)