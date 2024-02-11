# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from label_anything.data.utils import get_preprocess_shape
from label_anything.utils.utils import ResultDict


class Dummy(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_size: int = 1024,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.rgb_conv = nn.Conv2d(3, 1, 3)
        self.emb_conv = nn.Conv2d(256, 1, 3)
        self.param = nn.Embedding(1, 256)

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
        if "prompt_points" in batched_input and batched_input["prompt_points"].flatten().shape[0] > 0:
            b, m, c, n, _ = batched_input["prompt_points"].shape
            device = batched_input["prompt_points"].device
            class_example_embeddings = batched_input["prompt_points"].mean(dim=(3, 4)).unsqueeze(3)
            class_example_embeddings = class_example_embeddings @ self.param.weight
        elif "prompt_bboxes" in batched_input and batched_input["prompt_bboxes"].flatten().shape[0] > 0:
            b, m, c, n, _ = batched_input["prompt_bboxes"].shape
            device = batched_input["prompt_bboxes"].device
            class_example_embeddings = batched_input["prompt_bboxes"].mean(dim=(3, 4)).unsqueeze(3)
            class_example_embeddings = class_example_embeddings @ self.param.weight
        elif "prompt_masks" in batched_input and batched_input["prompt_masks"].flatten().shape[0] > 0:
            b, m, c, h, w = batched_input["prompt_masks"].shape
            device = batched_input["prompt_masks"].device
            class_example_embeddings = batched_input["prompt_masks"].mean(dim=(3, 4)).unsqueeze(3)
            class_example_embeddings = class_example_embeddings @ self.param.weight
        
        
        if "embeddings" in batched_input:
            seg = self.emb_conv(batched_input["embeddings"][:, 0, ::]).repeat(1, c, 1, 1)
        else:
            seg = self.rgb_conv(batched_input["images"][:, 0, ::]).repeat(1, c, 1, 1)      

        seg = self.postprocess_masks(seg, batched_input["dims"])
        if "flag_gts" in batched_input:
            seg[batched_input["flag_gts"].logical_not()] = -1 * torch.inf
        return {
            ResultDict.LOGITS: seg,
            ResultDict.EXAMPLES_CLASS_EMBS: class_example_embeddings,
        }
        
    def generate_class_embeddings(self, example_dict, chunk_size: int = None) -> torch.Tensor:
        if "prompt_points" in example_dict and example_dict["prompt_points"].flatten().shape[0] > 0:
            b, m, c, n, _ = example_dict["prompt_points"].shape
            device = example_dict["prompt_points"].device
            class_example_embeddings = example_dict["prompt_points"].mean(dim=(3, 4)).unsqueeze(3)
            class_example_embeddings = class_example_embeddings @ self.param.weight
        elif "prompt_bboxes" in example_dict and example_dict["prompt_bboxes"].flatten().shape[0] > 0:
            b, m, c, n, _ = example_dict["prompt_bboxes"].shape
            device = example_dict["prompt_bboxes"].device
            class_example_embeddings = example_dict["prompt_bboxes"].mean(dim=(3, 4)).unsqueeze(3)
            class_example_embeddings = class_example_embeddings @ self.param.weight
        elif "prompt_masks" in example_dict and example_dict["prompt_masks"].flatten().shape[0] > 0:
            b, m, c, h, w = example_dict["prompt_masks"].shape
            device = example_dict["prompt_masks"].device
            class_example_embeddings = example_dict["prompt_masks"].mean(dim=(3, 4)).unsqueeze(3)
            class_example_embeddings = class_example_embeddings @ self.param.weight
        return class_example_embeddings.mean(dim=1)
    
    def predict(self, batched_input):
        c = self.class_embeddings.shape[1]
        if "embeddings" in batched_input:
            seg = self.emb_conv(batched_input["embeddings"]).repeat(1, c, 1, 1)
        else:
            seg = self.rgb_conv(batched_input["images"]).repeat(1, c, 1, 1)      

        seg = self.postprocess_masks(seg, batched_input["dims"])
        if "flag_gts" in batched_input:
            seg[batched_input["flag_gts"].logical_not()] = -1 * torch.inf
        return seg
        
    
    def get_learnable_params(self, training_params: dict) -> list:
        """

        :return: list of dictionaries containing the key 'named_params' with a list of named params
        """
        return self.parameters()

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
                    value=float("-inf"),
                )
                for i, mask in enumerate(masks)
            ]
        )
        masks[:, 0, :, :][masks[:, 0, :, :] == float("-inf")] = 0 # background class for padding
        return masks


def build_dummy() -> Dummy:
    return Dummy()