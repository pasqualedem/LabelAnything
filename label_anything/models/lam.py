# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from label_anything.data.utils import get_preprocess_shape

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
        image_size: int = 1024,
    ) -> None:
        """
        LAM predicts object masks from an image and a list of examples images with prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts with their corresponding images.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
        """
        super().__init__()
        self.image_size = image_size
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward(
        self, batched_input: List[Dict[str, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (dict): a dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'images': The query + N example images as a torch tensor in Bx(N+1)x3xHxW format,
                already transformed for input to the model.
              'embeddings': The query + N example embeddings as a torch tensor in Bx(N+1)NCxHxW format.
                In alternative to 'query_image', 'query_embedding' can be provided.
              'prompt_points': (torch.Tensor) Batched point prompts for
                this image, with shape BxMxCxNx2. Already transformed to the
                input frame of the model.
              'flag_points': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'prompt_bboxes': (torch.Tensor) Batched box inputs, with shape BxMxCx4.
                Already transformed to the input frame of the model.
              'box_flag': (torch.Tensor) Batched box flag, with shape BxMxC.
              'prompt_masks': (torch.Tensor) Batched mask inputs to the model,
                in the form BxMxCxHxW.s
              'flag_masks': (torch.Tensor) Batched mask flag to indicate which flag is valid, BxMxC

        Returns:
          torch.Tensor: Batched multiclass mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is the number of classes, (H, W) is the
                original size of the image.
        """
        query_embeddings, prompt_embeddings = self.prepare_query_example_embeddings(batched_input)
        points, boxes, masks = self.prepare_prompts(batched_input)

        class_embeddings = self.prompt_encoder(
            image_embeddings=prompt_embeddings,
            points=points,
            boxes=boxes,
            masks=masks,
        )

        seg = self.mask_decoder(
            image_embeddings=query_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            class_embeddings=class_embeddings,
        )

        return self.postprocess_masks(seg, batched_input["dims"])

    def prepare_query_example_embeddings(self, batched_input):
        if "embeddings" in batched_input:
            embeddings = batched_input["embeddings"]
            B, N, C, H, W = embeddings.shape
            query_embeddings = embeddings[:, 0]
            prompt_embeddings = embeddings[:, 1:]
        elif "images" in batched_input:
            B, N, C, H, W = batched_input["images"].shape
            images = rearrange(batched_input["images"], "b n c h w -> (b n) c h w")
            embeddings = rearrange(
                self.image_encoder(images), "(b n) c h w -> b n c h w", b=B
            )
            query_embeddings = embeddings[:, 0]
            prompt_embeddings = embeddings[:, 1:]
        else:
            raise ValueError("Either 'images' or 'embeddings' must be provided.")

        return query_embeddings, prompt_embeddings
    
    def prepare_embeddings(self, batched_input):
        if "embeddings" in batched_input:
            embeddings = batched_input["embeddings"]
            B, N, C, H, W = embeddings.shape
        elif "images" in batched_input:
            B, N, C, H, W = batched_input["images"].shape
            images = rearrange(batched_input["images"], "b n c h w -> (b n) c h w")
            embeddings = self.image_encoder(images)
        else:
            raise ValueError("Either 'images' or 'embeddings' must be provided.")

        return embeddings

    def prepare_prompts(self, batched_input):
        if "prompt_points" in batched_input:
            points = (batched_input["prompt_points"], batched_input["flag_points"])
        else:
            points = None
        if "prompt_bboxes" in batched_input:
            boxes = batched_input["prompt_bboxes"]
            box_flag = batched_input["flag_bboxes"]
            boxes = (boxes, box_flag)
        else:
            boxes = None
        if "prompt_masks" in batched_input:
            masks = (batched_input["prompt_masks"], batched_input["flag_masks"])
        else:
            masks = None

        return points, boxes, masks

    def init_pretrained_weights(self, weights):
        """
        Initialize certain modules with pretrained weights from Sam.
        """
        # Load weights for the image encoder
        if self.image_encoder is not None:
            image_encoder_weights = {
                k[len("image_encoder.") :]: v
                for k, v in weights.items()
                if k.startswith("image_encoder")
            }
            self.image_encoder.load_state_dict(image_encoder_weights)
        # Load weights for the prompt encoder
        pe_layer_weights = {
            k[len("prompt_encoder.pe_layer.") :]: v
            for k, v in weights.items()
            if k.startswith("prompt_encoder.pe_layer")
        }
        self.prompt_encoder.pe_layer.load_state_dict(pe_layer_weights)
        point_embeddings_weights = {
            k[len("prompt_encoder.point_embeddings.") :]: v
            for k, v in weights.items()
            if k.startswith("prompt_encoder.point_embeddings")
        }
        self.prompt_encoder.point_embeddings.load_state_dict(point_embeddings_weights)
        not_a_point_embed_weights = {
            k[len("prompt_encoder.not_a_point_embed.") :]: v
            for k, v in weights.items()
            if k.startswith("prompt_encoder.not_a_point_embed")
        }
        self.prompt_encoder.not_a_point_embed.load_state_dict(not_a_point_embed_weights)
        mask_downscaling_weights = {
            k[len("prompt_encoder.mask_downscaling.") :]: v
            for k, v in weights.items()
            if k.startswith("prompt_encoder.mask_downscaling")
        }
        self.prompt_encoder.mask_downscaling.load_state_dict(mask_downscaling_weights)
        no_mask_embed_weights = {
            k[len("prompt_encoder.no_mask_embed.") :]: v
            for k, v in weights.items()
            if k.startswith("prompt_encoder.no_mask_embed")
        }
        self.prompt_encoder.no_mask_embed.load_state_dict(no_mask_embed_weights)

        # Load tranformer weights
        transformer_weights = {
            k[len("mask_decoder.transformer.") :]: v
            for k, v in weights.items()
            if k.startswith("mask_decoder.transformer")
        }
        self.prompt_encoder.transformer.load_state_dict(transformer_weights)

        # Load weights for the mask decoder transformer
        self.mask_decoder.transformer.load_state_dict(transformer_weights.copy())

        # Load weights for the mask decoder output upscaling
        output_upscaling_weights = {
            k[len("mask_decoder.output_upscaling.") :]: v
            for k, v in weights.items()
            if k.startswith("mask_decoder.output_upscaling")
        }
        self.mask_decoder.output_upscaling.load_state_dict(output_upscaling_weights)

    def generate_class_embeddings(self, example_dict):
        prompt_embeddings = self.prepare_embeddings(example_dict)
        points, boxes, masks = self.prepare_prompts(example_dict)
        class_embeddings = self.prompt_encoder(
            image_embeddings=prompt_embeddings,
            points=points,
            boxes=boxes,
            masks=masks,
        )
        return class_embeddings
    
    def predict(self, batched_input, class_embeddings=None):
        if class_embeddings is None:
            return self.forward(batched_input)
        query_embeddings = self.prepare_embeddings(batched_input)
        
        seg = self.mask_decoder(
            image_embeddings=query_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            class_embeddings=class_embeddings,
        )

        return self.postprocess_masks(seg, batched_input["dims"])

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          original_size (torch.Tensor): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        max_original_size = torch.max(original_sizes.view(-1, 2), 0).values.tolist()
        original_sizes = original_sizes[:, 0, :]
        input_sizes = [
            get_preprocess_shape(h, w, self.image_size) for (h, w) in original_sizes
        ]
        masks = F.interpolate(
            masks,
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = [
            masks[i, :, : input_size[0], : input_size[1]]
            for i, input_size in enumerate(input_sizes)
        ]
        masks = [
            F.interpolate(
                torch.unsqueeze(masks[i], 0),
                original_size.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for i, original_size in enumerate(original_sizes)
        ]

        masks = torch.cat(
            [
                F.pad(
                    mask,
                    (
                        0,
                        max_original_size[1] - original_sizes[i, 1],
                        0,
                        max_original_size[0] - original_sizes[i, 0],
                    ),
                    mode="constant",
                    value=0.0,
                )
                for i, mask in enumerate(masks)
            ]
        )
        return masks
