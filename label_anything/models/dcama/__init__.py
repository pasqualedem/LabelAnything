import torch
import torch.nn.functional as F

from einops import rearrange, repeat

from label_anything.models.dcama.dcama import DCAMA
from label_anything.utils.utils import ResultDict, torch_dict_load
from label_anything.data.utils import BatchKeys
from label_anything.data.utils import get_preprocess_shape


def build_dcama(
    backbone: str = "swin",
    backbone_checkpoint: str = "checkpoints/backbone.pth",
    model_checkpoint: str = None,
    image_size: int = 384,
    custom_preprocess: bool = False,
):
    model = DCAMAMultiClass(
        backbone, backbone_checkpoint, use_original_imgsize=False, image_size=image_size
    )
    params = model.state_dict()
    if model_checkpoint is None:
        return model
    state_dict = torch_dict_load(model_checkpoint)

    if model_checkpoint.endswith(".pt"): # DCAMA original repo
        for k1, k2 in zip(list(state_dict.keys()), params.keys()):
            state_dict[k2] = state_dict.pop(k1)
    elif model_checkpoint.endswith(".safetensors"): # LA Repo
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print("Error loading state_dict, trying to load without 'model.' prefix")
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

    model.load_state_dict(state_dict)
    return model


class DCAMAMultiClass(DCAMA):
    def __init__(self, backbone, pretrained_path, use_original_imgsize, image_size):
        self.predict = None
        self.generate_class_embeddings = None
        self.image_size = image_size
        super().__init__(backbone, pretrained_path, use_original_imgsize)

    def _preprocess_masks(self, masks, dims):
        B, N, C, H, W = masks.size()
        # remove bg from masks
        masks = masks[:, :, 1:, ::]
        mask_size = 256

        # Repeat dims along class dimension
        support_dims = dims[:, 1:]
        repeated_dims = repeat(support_dims, "b n d -> (b n c) d", c=C)
        masks = rearrange(masks, "b n c h w -> (b n c) h w")

        # Remove padding from masks
        # pad_dims = [get_preprocess_shape(h, w, mask_size) for h, w in repeated_dims]
        # masks = [mask[:h, :w] for mask, (h, w) in zip(masks, pad_dims)]
        # masks = torch.cat(
        #     [
        #         F.interpolate(
        #             torch.unsqueeze(mask, 0).unsqueeze(0),
        #             size=(self.image_size, self.image_size),
        #             mode="nearest",
        #         )[0]
        #         for mask in masks
        #     ]
        # )
        return rearrange(masks, "(b n c) h w -> b n c h w", b=B, n=N)

    def forward(self, x):

        x[BatchKeys.PROMPT_MASKS] = self._preprocess_masks(
            x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS]
        )
        # assert (
        #     x[BatchKeys.PROMPT_MASKS].shape[0] == 1
        # ), "Only tested with batch size = 1"
        logits = []
        # get logits for each class
        B = x[BatchKeys.PROMPT_MASKS].shape[0]
        for c in range(x[BatchKeys.PROMPT_MASKS].size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = class_examples.sum(dim=1)
            assert (n_shots == n_shots[0]).all(), "Only support same number of examples for each class"
            n_shots = n_shots[0]
            class_input_dict = {
                BatchKeys.IMAGES: x[BatchKeys.IMAGES],
                BatchKeys.PROMPT_MASKS: rearrange(x[BatchKeys.PROMPT_MASKS][:, :, c, ::][class_examples], "(b k) h w -> b k h w", b=B),
            }
            logits.append(self.predict_mask_nshot(class_input_dict, n_shots))
        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = self.postprocess_masks(logits, x["dims"])

        return {
            ResultDict.LOGITS: logits,
        }

    def postprocess_masks(self, logits, dims):
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        dims = dims[:, 0, :]  # get real sizes of the query images
        logits = [
            F.interpolate(
                torch.unsqueeze(logit, 0),
                size=dim.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for logit, dim in zip(logits, dims)
        ]

        logits = torch.cat(
            [
                F.pad(
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

        # set padding to background class
        logits[:, 0, :, :][logits[:, 0, :, :] == float("-inf")] = 0
        return logits

    def get_learnable_params(self, train_params):
        return self.parameters()
