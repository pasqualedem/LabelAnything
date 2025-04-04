# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from label_anything.data.utils import BatchKeys, get_preprocess_shape
from label_anything.models.transformer import TwoWayTransformer
from label_anything.models.common import SAM_EMBED_DIM
from label_anything.utils.utils import ResultDict

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder, MultiLevelMaskDecoder
from .prompt_encoder import PromptImageEncoder, MultiLevelPromptEncoder


class Lam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptImageEncoder,
        mask_decoder: MaskDecoder,
        neck: nn.Module,
        image_size: int = 1024,
        custom_preprocess: bool = True,
    ) -> None:
        """
        LAM predicts object masks from an image and a list of examples images with prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts with their corresponding images.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          custom_preprocess (bool): Whether to use custom preprocessing (padding)
        """
        super().__init__()
        self.image_size = image_size
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.class_embeddings = None
        self.neck = neck
        self.custom_preprocess = custom_preprocess

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
        seg, pe_result = self._forward(batched_input)
        seg = self.postprocess_masks(seg, batched_input["dims"])
        if "flag_gts" in batched_input:
            seg[batched_input["flag_gts"].logical_not()] = -1 * torch.inf
        result = {
            ResultDict.LOGITS: seg,
            ResultDict.EXAMPLES_CLASS_EMBS: pe_result[ResultDict.EXAMPLES_CLASS_EMBS],
        }
        if ResultDict.MASK_EMBEDDINGS in pe_result:
            result[ResultDict.MASK_EMBEDDINGS] = pe_result[ResultDict.MASK_EMBEDDINGS]
        return result

    def get_dense_pe(self):
        if (
            hasattr(self.mask_decoder, "transformer_feature_size")
            and self.mask_decoder.transformer_feature_size is not None
        ):
            return self.prompt_encoder.pe_layer(
                (
                    self.mask_decoder.transformer_feature_size,
                    self.mask_decoder.transformer_feature_size,
                )
            ).unsqueeze(0)
        return self.prompt_encoder.get_dense_pe()

    def _forward(self, batched_input: List[Dict[str, Any]]) -> torch.Tensor:
        query_embeddings, prompt_embeddings = self.prepare_query_example_embeddings(
            batched_input
        )
        points, boxes, masks, flag_examples = self.prepare_prompts(batched_input)

        pe_result = self.prompt_encoder(
            image_embeddings=prompt_embeddings,
            points=points,
            boxes=boxes,
            masks=masks,
            flag_examples=flag_examples,
        )

        seg = self.mask_decoder(
            query_embeddings=query_embeddings,
            support_embeddings=prompt_embeddings,
            image_pe=self.get_dense_pe(),
            pe_result=pe_result,
            flag_examples=flag_examples,
        )
        return seg, pe_result

    def prepare_query_example_embeddings(self, batched_input):
        if "embeddings" in batched_input:
            embeddings = batched_input["embeddings"]
            if not isinstance(embeddings, dict):
                B, N, C, H, W = embeddings.shape
                if self.neck is not None:
                    embeddings = rearrange(embeddings, "b n c h w -> (b n) c h w", b=B)
                    embeddings = self.neck(embeddings)
                    embeddings = rearrange(embeddings, "(b n) c h w -> b n c h w", b=B)
            else:
                key0 = next(iter(embeddings.keys()))
                B, N, C, H, W = embeddings[key0].shape
                if self.neck is None:
                    raise ValueError("Feature pyramids require a PyramidNeck")
                embeddings = {
                    k: rearrange(v, "b n c h w -> (b n) c h w", b=B) for k, v in embeddings.items()
                }
                embeddings = self.neck(embeddings)
                embeddings = rearrange(embeddings, "(b n) c h w -> b n c h w", b=B)
        elif "images" in batched_input:
            B, N, C, H, W = batched_input["images"].shape
            images = rearrange(batched_input["images"], "b n c h w -> (b n) c h w")
            embeddings = self.image_encoder(images)
            if self.neck is not None:
                embeddings = self.neck(embeddings)
            embeddings = rearrange(embeddings, "(b n) c h w -> b n c h w", b=B)
        else:
            raise ValueError("Either 'images' or 'embeddings' must be provided.")

        query_embeddings = embeddings[:, 0]
        prompt_embeddings = embeddings[:, 1:]

        return query_embeddings, prompt_embeddings
    
    # Added to work with support images' embeddings only
    def prepare_embeddings_example(self, batched_input):
        if "embeddings" in batched_input:
            embeddings = batched_input["embeddings"]
            B, N, C, H, W = embeddings.shape
            if self.neck is not None:
                embeddings = rearrange(embeddings, "b n c h w -> (b n) c h w", b=B)
                embeddings = self.neck(embeddings)
                embeddings = rearrange(embeddings, "(b n) c h w -> b n c h w", b=B)
        elif "images" in batched_input:
            B, N, C, H, W = batched_input["images"].shape
            images = rearrange(batched_input["images"], "b n c h w -> (b n) c h w")
            embeddings = self.image_encoder(images)
            if self.neck is not None:
                embeddings = self.neck(embeddings)
            embeddings = rearrange(embeddings, "(b n) c h w -> b n c h w", b=B)
        else:
            raise ValueError("Either 'images' or 'embeddings' must be provided.")
        return embeddings

    def prepare_embeddings(self, batched_input, chunk_size=None):
        if "embeddings" in batched_input:
            embeddings = batched_input["embeddings"]
        elif "images" in batched_input:
            images = batched_input["images"]
            B, N = images.shape[0:2]
            images = rearrange(images, "b n c h w -> (b n) c h w")
            if chunk_size is not None:
                embeddings = []
                for i in range(0, N, chunk_size):
                    embeddings.append(self.image_encoder(images[i : i + chunk_size]))
                embeddings = torch.cat(embeddings, dim=0)
            else:
                embeddings = self.image_encoder(images)
            if self.neck is not None:
                embeddings = self.neck(embeddings)
            embeddings = rearrange(embeddings, "(b n) c h w -> b n c h w", b=B)
        else:
            raise ValueError("Either 'images' or 'embeddings' must be provided.")

        return embeddings

    def prepare_prompts(self, batched_input):
        if (
            "prompt_points" in batched_input
            and (batched_input["flag_points"] == 0).all().logical_not()
        ):
            points = (batched_input["prompt_points"], batched_input["flag_points"])
        else:
            points = None
        if (
            "prompt_bboxes" in batched_input
            and (batched_input["flag_bboxes"] == 0).all().logical_not()
        ):
            boxes = batched_input["prompt_bboxes"]
            box_flag = batched_input["flag_bboxes"]
            boxes = (boxes, box_flag)
        else:
            boxes = None
        if (
            "prompt_masks" in batched_input
            and (batched_input["flag_masks"] == 0).all().logical_not()
        ):
            masks = (batched_input["prompt_masks"], batched_input["flag_masks"])
        else:
            masks = None

        return points, boxes, masks, batched_input["flag_examples"]

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
        if (
            self.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.shape[1]
            == 2 * SAM_EMBED_DIM
        ):
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
            self.prompt_encoder.point_embeddings.load_state_dict(
                point_embeddings_weights
            )
            not_a_point_embed_weights = {
                k[len("prompt_encoder.not_a_point_embed.") :]: v
                for k, v in weights.items()
                if k.startswith("prompt_encoder.not_a_point_embed")
            }
            self.prompt_encoder.not_a_point_embed.load_state_dict(
                not_a_point_embed_weights
            )
            mask_downscaling_weights = {
                k[len("prompt_encoder.mask_downscaling.") :]: v
                for k, v in weights.items()
                if k.startswith("prompt_encoder.mask_downscaling")
            }
            self.prompt_encoder.mask_downscaling.load_state_dict(
                mask_downscaling_weights
            )
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
            if self.prompt_encoder.transformer.attention_downsample_rate == 2:
                self.prompt_encoder.transformer.load_state_dict(transformer_weights)

            # Load weights for the mask decoder transformer
            if (
                isinstance(self.mask_decoder.transformer, TwoWayTransformer)
                and self.mask_decoder.transformer.attention_downsample_rate == 2
            ):
                self.mask_decoder.transformer.load_state_dict(
                    transformer_weights.copy()
                )

            # Load weights for the mask decoder output upscaling
            output_upscaling_weights = {
                k[len("mask_decoder.output_upscaling.") :]: v
                for k, v in weights.items()
                if k.startswith("mask_decoder.output_upscaling")
            }
            self.mask_decoder.output_upscaling.load_state_dict(output_upscaling_weights)

    def get_learnable_params(self, training_params: dict) -> list:
        """

        :return: list of dictionaries containing the key 'named_params' with a list of named params
        """

        def f(x):
            return not "image_encoder" in x[0]

        freeze_pretrained = training_params.get("freeze_backbone", False)
        if freeze_pretrained and "backbone_lr" in training_params:
            raise ValueError(
                "Cannot freeze the backbone and set a learning rate for it at the same time."
            )
        if freeze_pretrained:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            return [x[1] for x in filter(f, list(self.named_parameters()))]
        if "backbone_lr" in training_params:
            return [
                {
                    "params": self.image_encoder.parameters(),
                    "lr": training_params["backbone_lr"],
                },
                {"params": [x[1] for x in filter(f, list(self.named_parameters()))]},
            ]
        return self.parameters()

    def generate_class_embeddings(self, example_dict, chunk_size=None):
        prompt_embeddings = self.prepare_embeddings(example_dict, chunk_size=chunk_size)
        points, boxes, masks, flag_examples = self.prepare_prompts(example_dict)
        class_embeddings = self.prompt_encoder(
            image_embeddings=prompt_embeddings,
            points=points,
            boxes=boxes,
            masks=masks,
            chunk_size=chunk_size,
            flag_examples=flag_examples,
        )
        return class_embeddings

    def predict(self, batched_input, class_embeddings=None):
        if class_embeddings is None and self.class_embeddings is None:
            return self.forward(batched_input)
        if class_embeddings is None and self.class_embeddings is not None:
            class_embeddings = self.class_embeddings
        query_embeddings = self.prepare_embeddings(batched_input)[
            :, 0
        ]  # There is only query image

        seg = self.mask_decoder(
            query_embeddings=query_embeddings,
            support_embeddings=None,
            image_pe=self.prompt_encoder.get_dense_pe(),
            pe_result=class_embeddings,
            flag_examples=None,
        )

        return self.postprocess_masks(
            seg, batched_input["dims"].unsqueeze(1)  # Add example dimension to uniform
        )

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
        original_sizes = original_sizes[:, 0, :]  # get real sizes of the query images
        input_sizes = [
            get_preprocess_shape(h, w, self.image_size) for (h, w) in original_sizes
        ]  # these are the input sizes without padding

        # interpolate masks to the model size
        masks = F.interpolate(
            masks,
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        if self.custom_preprocess:
            # remove padding from masks
            masks = [
                masks[i, :, : input_size[0], : input_size[1]]
                for i, input_size in enumerate(input_sizes)
            ]
            
        # interpolate masks to the original size
        masks = [
            F.interpolate(
                torch.unsqueeze(masks[i], 0),
                original_size.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for i, original_size in enumerate(original_sizes)
        ]

        # pad masks to the same size, use -inf so they don't affect the softmax
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
                    value=float("-inf"),
                )
                for i, mask in enumerate(masks)
            ]
        )

        # set padding to background class
        masks[:, 0, :, :][masks[:, 0, :, :] == float("-inf")] = 0
        return masks


class BinaryLam(Lam):
    def _build_class_dict(self, x, c):
        class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c]
        prompt_keys = [
            BatchKeys.PROMPT_MASKS,
            BatchKeys.PROMPT_BBOXES,
            BatchKeys.PROMPT_POINTS,
        ]
        flag_keys = [
            BatchKeys.FLAG_MASKS,
            BatchKeys.FLAG_BBOXES,
            BatchKeys.FLAG_POINTS,
            BatchKeys.FLAG_EXAMPLES,
        ]
        prompt_input_dict = {
            key: torch.cat(
                [
                    x[key][:, :, 0, ::][class_examples].unsqueeze(1).unsqueeze(0),
                    x[key][:, :, c, ::][class_examples].unsqueeze(1).unsqueeze(0),
                ],
                dim=2,
            )
            for key in prompt_keys
        }
        flag_input_dict = {
            key: torch.cat(
                [
                    x[key][:, :, 0][class_examples].unsqueeze(1).unsqueeze(0),
                    x[key][:, :, c][class_examples].unsqueeze(1).unsqueeze(0),
                ],
                dim=2,
            )
            for key in flag_keys
        }
        class_input_dict = {
            BatchKeys.IMAGES: torch.cat(
                [x[BatchKeys.IMAGES][:, 0], x[BatchKeys.IMAGES][:, 1:][class_examples]]
            ).unsqueeze(0),
            **prompt_input_dict,
            **flag_input_dict,
        }
        return class_input_dict

    def forward(self, x: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        B, M, C = x[BatchKeys.FLAG_EXAMPLES].shape
        assert (
            x[BatchKeys.FLAG_EXAMPLES].shape[0] == 1
        ), "Only tested with batch size = 1"
        results = []
        # get logits for each class
        for c in range(1, x[BatchKeys.FLAG_EXAMPLES].size(2)):
            class_input_dict = self._build_class_dict(x, c)
            results.append(super()._forward(class_input_dict))
        logits, embeddings = zip(*results)
        dummy_embeddings = torch.zeros(
            (B, M, C, embeddings[0][ResultDict.EXAMPLES_CLASS_EMBS].shape[-1]),
            device=embeddings[0][ResultDict.EXAMPLES_CLASS_EMBS].device,
        )

        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = self.postprocess_masks(logits, x["dims"])
        logits[x["flag_gts"].logical_not()] = -1 * torch.inf

        return {
            ResultDict.LOGITS: logits,
            ResultDict.EXAMPLES_CLASS_EMBS: dummy_embeddings,
        }


class MultiLevelLam(Lam):
    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: MultiLevelPromptEncoder,
        mask_decoder: MultiLevelMaskDecoder,
        neck: nn.Module,
        image_size: int = 1024,
    ) -> None:
        super().__init__(image_encoder, prompt_encoder, mask_decoder, neck, image_size)
        
    def _forward_encoder(self, images: List[Dict[str, Any]]) -> torch.Tensor:
        B, N = images.shape[0:2]
        images = rearrange(images, "b n c h w -> (b n) c h w")
        embeddings = self.image_encoder(
            pixel_values=images, output_hidden_states=True
        )["hidden_states"]
        embeddings = [
            rearrange(embedding, "(b n) c h w -> b n c h w", b=B)
            for embedding in embeddings
        ]
        return embeddings
        
    def prepare_query_example_embeddings(self, batched_input):
        if "embeddings" in batched_input:
            embeddings = batched_input["embeddings"]
            B, N, C, H, W = embeddings.shape
            if self.neck is not None:
                raise NotImplementedError("Neck not implemented for MultiLevelLam")
        elif "images" in batched_input:
            images = batched_input["images"]
            embeddings = self._forward_encoder(images)
            if self.neck is not None:
                raise NotImplementedError("Neck not implemented for MultiLevelLam")
        else:
            raise ValueError("Either 'images' or 'embeddings' must be provided.")

        query_embeddings = [embedding[:, 0] for embedding in embeddings]
        prompt_embeddings = [embedding[:, 1:] for embedding in embeddings]

        return query_embeddings, prompt_embeddings

    def prepare_embeddings(self, batched_input):
        if "embeddings" in batched_input:
            embeddings = batched_input["embeddings"]
        elif "images" in batched_input:
            images = batched_input["images"]
            embeddings = self._forward_encoder(images)
        else:
            raise ValueError("Either 'images' or 'embeddings' must be provided.")

        return embeddings
