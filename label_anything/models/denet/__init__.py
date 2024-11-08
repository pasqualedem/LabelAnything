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


def remove_duplicates_by_frequency(classes, flag_examples):
    # Count occurrences of each element across all sublists
    element_counts = Counter(element for sublist in classes for element in sublist)

    # Result list to store modified sublists
    result = []

    for sublist, flag_example in zip(classes, flag_examples):
        # If there's more than one element in the sublist
        if len(sublist) > 1:
            # Sort elements by the frequency count, keeping the less frequent one
            ssublist = sorted(sublist, key=lambda x: element_counts[x])
            to_keep = ssublist[0]
            element_counts[to_keep] += 1
            to_not_keep = ssublist[1:]
            for to_not_keep_elem in to_not_keep:
                to_not_keep_idx = sublist.index(to_not_keep_elem)
                remove_idx = torch.where(flag_example)[0][to_not_keep_idx + 1]
                flag_example[remove_idx] = 0
            # Keep only the least frequent element
            result.append([to_keep])
        else:
            # If it's a single element, keep it as is
            result.append(sublist)

    return result


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
        classes = [
            remove_duplicates_by_frequency(class_item[1:], flag_item)
            for class_item, flag_item in zip(batch["classes"], flag_examples)
        ]
        classes = torch.tensor(classes).to(Ys.device)
        flag_examples = torch.tensor(flag_examples).to(Ys.device)
        label = classes.reshape(b, c_fg, k)[:, :, 0]
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
