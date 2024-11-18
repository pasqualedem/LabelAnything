import torch.nn.functional as Ft

from collections import Counter
from einops import rearrange
from label_anything.data.utils import BatchKeys
from label_anything.utils.utils import ResultDict, torch_dict_load
from .backbone import *
from .head import *
from .common import *

from . import DENet as OriginalDENet

def nested_stack(nested_list):
    # Base case: If the innermost element is a tensor, we can return directly
    if isinstance(nested_list[0], torch.Tensor):
        return torch.stack(nested_list)
    
    # Recursive case: Apply nested_stack to each sublist
    return torch.stack([nested_stack(sublist) for sublist in nested_list])


def remove_duplicated_classes(classes, intended_classes, flag_examples):

    all_classes = sorted(set.union(*[set(sublist) for sublist in classes]))
    class_to_flag = {cls: idx + 1 for idx, cls in enumerate(all_classes)}

    for sublist_c, sublist_ic, flag_example in zip(classes, intended_classes, flag_examples):
        additional_classes = set(sublist_c) - set(sublist_ic)
        for cls in additional_classes:
            cls_idx = class_to_flag[cls]
            flag_example[cls_idx] = 0
    assert (
        len(flag_examples.sum(dim=1).unique()) == 1
    ), "There are classes with different occurrences"
    assert (
        len(flag_examples[:, 1:].sum(dim=0).unique()) == 1
    ), "There are examples with different number of classes"

class DeNet(OriginalDENet):
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
    def forward(self, batch: dict):
        images = batch[BatchKeys.IMAGES]
        Iq = images[:, 0]
        Is = images[:, 1:]
        Ys = batch[BatchKeys.PROMPT_MASKS]  # B M C H W
        b, m, c, _, _ = Ys.shape
        c_fg = c - 1
        k = m // c_fg
        flag_examples = batch[BatchKeys.FLAG_EXAMPLES].clone()
        intended_classes = batch[BatchKeys.INTENDED_CLASSES]
        classes = batch[BatchKeys.CLASSES]
        for cls, intended_class, flag_example in zip(classes, intended_classes, flag_examples):
            remove_duplicated_classes(cls[1:], intended_class[1:], flag_example)
        intended_classes = torch.tensor(intended_classes).to(Ys.device)
        label = intended_classes[:, 1:] # remove query image
        label = rearrange(label, "b c 1 -> b c")
        flag_examples = torch.tensor(flag_examples).to(Ys.device)
        flag_examples[:, :, 0] = 0
        # Ys = rearrange(Ys[flag_examples], "(b k c) ... -> b c k ...", b=b, k=k, c=c_fg)
        # Is: (B, way, shot, 3, H, W)
        Y_list = [[[] for _ in range(c_fg)] for _ in range(b)]
        X_list = [[[] for _ in range(c_fg)] for _ in range(b)]
        # Is = rearrange(Is, "b (k c) rgb h w -> b c k rgb h w", k=k)
        for i, elem in enumerate(flag_examples):
            elem = elem[:, 1:]
            for j, shot in enumerate(elem):
                idx = torch.where(shot)[0][0]
                Y_list[i][idx].append(Ys[i, j, idx+1])
                X_list[i][idx].append(Is[i, j])
        Y_list = nested_stack(Y_list)
        X_list = nested_stack(X_list)
        if c_fg == 1:
            label = label.squeeze(-1)
        out = super().forward(X_list, Y_list, Iq, label)
        fb_logits = rearrange(out[1], "(b c) fb h w -> b c fb h w", c=c_fg)
        bg_logits = fb_logits[:, :, 0, ...]
        fg_logits = fb_logits[:, :, 1, ...]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)
        
        
        logits = self.postprocess_masks(logits, batch[BatchKeys.DIMS])
        return {
            ResultDict.LOGITS: logits,
        }


def build_denet(checkpoint, maximum_num_classes=21, custom_preprocess=True):
    model = DeNet(maximum_num_classes=maximum_num_classes)
    state_dict = torch_dict_load(checkpoint)
    model.load_state_dict(state_dict["model"])
    return model
