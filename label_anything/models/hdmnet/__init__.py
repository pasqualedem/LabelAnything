from collections import Counter
import os
from easydict import EasyDict

import torch
import torch.nn.functional as F

from label_anything.data.utils import BatchKeys
from label_anything.utils.utils import ResultDict
from .HDMNet import OneModel


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
    

class HDMNetModel(OneModel):
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
        return logits

    def forward(self, batch: dict):
        # remove bg from masks
        masks = batch[BatchKeys.PROMPT_MASKS][:, :, 1:, ::]
        y_m = None
        y_b = None
        cat_idx = None
        logits = []
        # get logits for each class
        flag_examples = batch[BatchKeys.FLAG_EXAMPLES].clone()

        for class_item, intended_class_item, flag_item in zip(
            batch[BatchKeys.CLASSES], batch[BatchKeys.INTENDED_CLASSES], flag_examples
        ):
            remove_duplicated_classes(class_item[1:], intended_class_item[1:], flag_item)

        for c in range(masks.size(2)):
            class_examples = flag_examples[:, :, c + 1]
            x = batch[BatchKeys.IMAGES][:, 0]
            s_x = batch[BatchKeys.IMAGES][:, 1:][class_examples].unsqueeze(0)
            s_y = masks[:, :, c, ::][class_examples].unsqueeze(0)
            n_shots = class_examples.sum().item()
            if (
                n_shots < self.shot
            ):  # if n_shots < self.shot repeat the last image and mask
                s_x = torch.cat(
                    [
                        s_x,
                        s_x[:, -1].unsqueeze(0).repeat(1, self.shot - n_shots, 1, 1, 1),
                    ],
                    dim=1,
                )
                s_y = torch.cat(
                    [s_y, s_y[:, -1].unsqueeze(0).repeat(1, self.shot - n_shots, 1, 1)],
                    dim=1,
                )
            logits.append(
                super().forward(x, s_x=s_x, s_y=s_y, y_m=y_m, y_b=y_b, cat_idx=cat_idx)
            )
        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)
        logits = self.postprocess_masks(logits, batch["dims"])

        return {
            ResultDict.LOGITS: logits,
        }


def build_hdmnet(dataset, shots=1, val_fold_idx=0, custom_preprocess=True):
    args = EasyDict(
        {
            "layers": 50,
            "vgg": False,
            "aux_weight1": 1.0,
            "aux_weight2": 1.0,
            "low_fea": "layer2",  # low_fea for computing the Gram matrix
            "kshot_trans_dim": 2,  # K-shot dimensionality reduction
            "merge": "final",  # fusion scheme for GFSS ('base' Eq(S1) | 'final' Eq(18) )
            "merge_tau": 0.9,  # fusion threshold tau
            "zoom_factor": 8,
            "shot": shots,
            "data_set": dataset,
            "ignore_label": 255,
            "print_freq": 10,
            "split": val_fold_idx,
        }
    )
    model = HDMNetModel(args, cls_type="Base")
    if dataset == "pascsal":
        raise NotImplementedError("PASCAL dataset is not supported yet")

    checkpoint_per_fold_1shot = {
        0: "checkpoints/hdmnet/coco/split0/resnet50/best_model.pth",
        1: "checkpoints/hdmnet/coco/split1/resnet50/best_model.pth",
        2: "checkpoints/hdmnet/coco/split2/resnet50/best_model.pth",
        3: "checkpoints/hdmnet/coco/split3/resnet50/best_model.pth",
    }

    checkpoint_per_fold_5shot = {
        0: "checkpoints/hdmnet/coco/split0/resnet50/best_model_5shot.pth",
        1: "checkpoints/hdmnet/coco/split1/resnet50/best_model_5shot.pth",
        2: "checkpoints/hdmnet/coco/split2/resnet50/best_model_5shot.pth",
        3: "checkpoints/hdmnet/coco/split3/resnet50/best_model_5shot.pth",
    }
    assert shots in [1, 5]
    checkpoint_per_fold = (
        checkpoint_per_fold_1shot if shots == 1 else checkpoint_per_fold_5shot
    )
    checkpoint_path = checkpoint_per_fold[val_fold_idx]

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        new_param = checkpoint["state_dict"]
        try:
            model.load_state_dict(new_param)
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            model.load_state_dict(new_param)
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return model
