import os
import wandb
import random
from einops import rearrange
import imageio
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, get_peft_model
from copy import deepcopy
import torchvision
from transformers import ViTMAEForPreTraining
from tqdm import tqdm
import yaml
from label_anything.loss import LabelAnythingLoss
from label_anything.utils.metrics import (
    DistributedMulticlassJaccardIndex,
    to_global_multiclass,
)
from label_anything.data import get_dataloaders
from label_anything.models import model_registry
from label_anything.utils.utils import torch_dict_load
from torch.optim import AdamW

import lovely_tensors as lt
import torch

from lora.substitutor import Substitutor, IncrementalSubstitutor
from lora.utils import (
    create_rgb_segmentation,
    print_trainable_parameters,
    random_foldername,
)

lt.monkey_patch()


substitutor_cls = {
    "default": Substitutor,
    "incremental": IncrementalSubstitutor,
}


DATASET_NAME = "val_coco20i"
dataset_args = {
    "datasets": {
        DATASET_NAME: {
            "name": "coco",
            "instances_path": "data/coco/annotations/instances_val2014.json",
            "emb_dir": "data/coco/vit_b_sam_embeddings/last_block_state",
            "img_dir": "data/coco/train_val_2017",
            "split": "val",
            "val_fold_idx": 3,
            "n_folds": 4,
            "n_shots": 5,
            "n_ways": 2,
            "do_subsample": False,
            "add_box_noise": False,
            "val_num_samples": 100,
        },
    },
    "common": {
        "remove_small_annotations": True,
        "do_subsample": False,
        "add_box_noise": True,
        "max_points_annotations": 70,
        "max_points_per_annotation": 10,
        "load_gts": False,
        "image_size": 480,
        "load_embeddings": False,
        "custom_preprocess": False,
    },
}

dataloader_args = {
    "num_workers": 0,
    "possible_batch_example_nums": [[1, 2, 4]],
    "val_possible_batch_example_nums": [[1, 1]],
    "prompt_types": ["mask"],
    "prompt_choice_level": ["episode"],
    "val_prompt_types": ["mask"],
}

model_params = {
    "class_attention": True,
    "example_class_attention": True,
    "class_encoder": {
        "bank_size": 100,
        "embed_dim": 256,
        "name": "RandomMatrixEncoder",
    },
    "embed_dim": 256,
    "example_attention": True,
    "example_class_attention": True,
    "fusion_transformer": "TwoWayTransformer",
    "image_embed_dim": 768,
    "image_size": 480,
    "spatial_convs": 3,
    "use_vit_sam_neck": False,
    "custom_preprocess": False,
}

name = "lam_mae_b"
path = "offline/wandb/generated-run-y04k97k7/files/best/model.safetensors"


class ViTModelWrapper(ViTMAEForPreTraining):
    def forward(self, x):
        h, w = x.shape[-2:]
        output = super().forward(x, interpolate_pos_encoding=True)
        hs = output.last_hidden_state[:, 1:, :]
        return rearrange(hs, "b (h w) c -> b c h w", h=h // 16).contiguous()

    def mae_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class LoraEvaluator:
    def __init__(
        self,
        model,
        dataloader,
        lora_config,
        num_iterations,
        lr,
        substitutor,
        print_folder,
        print_every=50,
        device="cuda",
        run=None,
    ):
        self.num_iterations = num_iterations
        self.lora_config = lora_config
        self.dataloader = dataloader
        self.print_every = print_every
        self.print_folder = print_folder
        self.device = device
        self.num_iterations = num_iterations
        self.lr = lr
        self.run = run

        os.makedirs(print_folder, exist_ok=True)
        self.dataset_categories = next(
            iter(dataloader.dataset.datasets.values())
        ).categories

        self.substitutor = substitutor
        self.model = model
        self.lora_model = get_peft_model(deepcopy(model), lora_config)
        print_trainable_parameters(self.lora_model)
        self.loss = LabelAnythingLoss(
            **{"class_weighting": True, "components": {"focal": {"weight": 1.0}}}
        )
        self.optimizer = AdamW(self.lora_model.parameters(), lr=lr)
        self.mious = [
            DistributedMulticlassJaccardIndex(
                num_classes=80 + 1,
                average="macro",
                ignore_index=-100,
            ).to(device)
            for _ in range(num_iterations)
        ]

    def reset_lora(self):
        self.lora_model = get_peft_model(deepcopy(self.model), self.lora_config)
        self.optimizer = AdamW(self.lora_model.parameters(), lr=self.lr)
        self.lora_model.to(self.device)

    def lora_step(self, batch_tuple, gt, bar):
        segmentation_preds = []
        for k in range(self.num_iterations):
            bar.set_description(
                f"Iteration {k} gpu memory: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            )
            self.substitutor.reset(batch=batch_tuple)
            for i, (batch, gt) in enumerate(self.substitutor):
                self.optimizer.zero_grad()
                if i == 0:
                    with torch.no_grad():
                        res = self.lora_model(batch)
                        loss_value = self.loss(res, gt)
                else:
                    res = self.lora_model(batch)
                    loss_value = self.loss(res, gt)
                    loss_value.backward()
                    self.optimizer.step()
                preds = res["logits"].argmax(dim=1)
                glob_preds, glob_gt = to_global_multiclass(
                    batch["classes"], self.dataset_categories, preds, gt
                )
                if i == 0:
                    self.mious[k].update(glob_preds, glob_gt)
                    segmentation_preds.append(preds.detach().cpu())
            # clear memory
            # del res, loss_value, preds, glob_preds, glob_gt
            # torch.cuda.empty_cache()
            # gc.collect()
        return segmentation_preds

    def print_results(self, i, batch_tuple, segmentation_preds):
        outfolder = f"{self.print_folder}/sample_{i}"
        os.makedirs(outfolder, exist_ok=True)
        segmentation_gts = [
            create_rgb_segmentation(batch_tuple[1][:, i].cpu())
            for i in range(batch_tuple[1].shape[1])
        ]
        segmentation_preds = [
            create_rgb_segmentation(pred) for pred in segmentation_preds
        ]
        resize_images = torchvision.transforms.functional.resize(
            batch_tuple[0]["images"][0], segmentation_gts[0].shape[2:]
        )
        plotted_images = torch.cat(
            [resize_images.cpu(), torch.cat(segmentation_gts)], dim=3
        )
        plotted_images.rgb.fig.savefig(f"{outfolder}/input_gt.png")
        for j, (segmentation_pred, segmentation_gt) in enumerate(
            zip(segmentation_preds, segmentation_gts)
        ):
            segmentation_pred.rgb.fig.savefig(f"{outfolder}/pred_{j}.png")

        # Create a gif from the generated segmentations
        frame_duration = 0.5
        images = [
            imageio.imread(f"{outfolder}/pred_{i}.png")
            for i in range(self.num_iterations)
        ]
        imageio.mimsave(
            f"{outfolder}/segmentation.gif",
            images,
            duration=frame_duration * len(images),
        )

    def print_mious(self):
        print("Printing mious")
        miou_values = [miou.compute().item() for miou in self.mious]
        for i, miou_value in enumerate(miou_values):
            if i == 0:
                self.run.log({"miou_orig": miou_value})
            else:
                self.run.log({f"miou_it_{i}": miou_value})
                self.run.log({f"gain_it_{i}": miou_value - miou_values[0]})
                self.run.log({"miou": miou_value})
                self.run.log({"gain": miou_value - miou_values[0]})
            print(f"Iteration {i}: miou: {miou_value}")
        plt.plot(miou_values)
        best_miou = max(miou_values)
        self.run.log({"best_miou": best_miou})
        best_gain = best_miou - miou_values[0]
        self.run.log({"best_gain": best_gain})

        plt.savefig(f"{self.print_folder}/mious.png")
        with open(f"{self.print_folder}/mious.txt", "w") as f:
            f.write("\n".join([str(m) for m in miou_values]))

    def evaluate(self):
        bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for i, (batch_tuple, data_name) in bar:
            self.reset_lora()
            batch_gt = batch_tuple[1].to(self.device)
            batch_dict = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_tuple[0].items()
            }
            batch_tuple = (batch_dict, batch_gt)
            segmentation_preds = self.lora_step(batch_tuple, batch_gt, bar)
            if i % self.print_every == 0:
                self.print_results(i, batch_tuple, segmentation_preds)
            bar.set_postfix(
                miou0=self.mious[0].compute().item(),
                miouN=self.mious[-1].compute().item(),
            )
        self.print_mious()


def main(params):
    foldername = random_foldername()

    # Set all the seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Numpy random seed
    np.random.seed(seed)
    # Python random seed
    random.seed(seed)
    
    num_iterations = params.get("num_iterations", 10)
    device = params.get("device", "cuda")
    lora_r = params.get("lora_r", 32)
    lora_alpha = params.get("lora_alpha", 32.0)
    lr = params.get("lr", 1e-4)
    target_modules = params.get("target_modules", ["query", "value"])
    lora_dropout = params.get("lora_dropout", 0.1)
    substitutor = params.get("substitutor", "default")
    n_ways = params.get("n_ways", 2)
    k_shots = params.get("k_shots", 5)
    
    dataset_args["datasets"][DATASET_NAME]["n_ways"] = n_ways
    dataset_args["datasets"][DATASET_NAME]["n_shots"] = k_shots
    train, val_dict, test = get_dataloaders(
        dataset_args, dataloader_args, num_processes=1
    )
    val = val_dict[DATASET_NAME]
    model = model_registry[name](**model_params)
    weights = torch_dict_load(path)
    weights = {k[6:]: v for k, v in weights.items()}

    keys = model.load_state_dict(weights, strict=False)
    for key in keys.missing_keys:
        if key.startswith("image_encoder"):
            continue
        print(f"Missing key: {key}")
    for key in keys.unexpected_keys:
        print(f"Unexpected key: {key}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        # modules_to_save=["classifier"],
    )

    folder = "offline"
    os.makedirs(folder, exist_ok=True)
    folder = "offline/lora"
    os.makedirs(folder, exist_ok=True)
    # Create a subfolder with the current time
    subfolder = f"{folder}/{foldername}"
    os.makedirs(subfolder, exist_ok=True)

    # Print params as yaml
    with open(f"{subfolder}/params.yaml", "w") as f:
        yaml.dump(params, f)

    run = wandb.init(project="lorafss", config=params)

    substitutor = substitutor_cls[substitutor](
            substitute=True,
            long_side_length=480,
            custom_preprocess=False,
            n_ways=n_ways,
            k_shots=k_shots,
        )

    lora_evaluator = LoraEvaluator(
        model,
        val,
        lora_config,
        num_iterations,
        lr,
        print_folder=subfolder,
        device=device,
        run=run,
        substitutor=substitutor,
    )
    lora_evaluator.evaluate()
    run.finish()
