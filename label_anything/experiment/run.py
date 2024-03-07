import copy
import json
import os
import random
import subprocess
import sys
import shutil
import uuid
from copy import deepcopy

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.optim import AdamW
from torchmetrics import MetricCollection
from tqdm import tqdm

from label_anything.data import get_dataloaders
from label_anything.data.utils import BatchKeys
from label_anything.experiment.substitution import Substitutor
from label_anything.experiment.utils import WrapperModule
from label_anything.logger.text_logger import get_logger
from label_anything.loss import LabelAnythingLoss
from label_anything.models import model_registry
from label_anything.utils.metrics import (
    DistributedBinaryJaccardIndex,
    DistributedMulticlassJaccardIndex,
    to_global_multiclass,
)
from label_anything.utils.utils import (
    FLOAT_PRECISIONS,
    ResultDict,
    RunningAverage,
    get_timestamp,
    write_yaml,
)

from .utils import (
    SchedulerStepMoment,
    allocate_memory,
    check_nan,
    compose_loss_input,
    get_batch_size,
    get_experiment_logger,
    get_scheduler,
    handle_oom,
    nosync_accumulation,
    parse_params,
    set_class_embeddings,
)
from label_anything.models.contrastive_pe import ContrastivePromptEncoder
from copy import deepcopy

logger = get_logger(__name__)
SIZE = 1024


class Run:
    def __init__(self):
        self.kd = None
        self.params = None
        self.dataset = None
        self.experiment = None
        self.plat_logger = None
        self.dataset_params = None
        self.train_params = None
        self.model = None
        self.scheduler = None
        self.criterion = None
        self.oom = None
        self.best_metric = None
        self.scheduler_step_moment = None
        self.watch_metric = None
        if "." not in sys.path:
            sys.path.extend(".")
        self.global_train_step = 0
        self.global_val_step = 0
        self.validation_json = None

    def parse_params(self, params: dict):
        self.params = deepcopy(params)

        (
            self.train_params,
            self.dataset_params,
            self.dataloader_params,
            self.model_params,
            self.prompt_encoder_params,
        ) = parse_params(self.params)

    def _load_prompt_encoder_parameters(self):
        if not self.prompt_encoder_params or self.model is None:
            return
        pe_params = deepcopy(self.prompt_encoder_params)
        pe_params["params"]["prompt_encoder"] = self.model.prompt_encoder
        contrastive_prompt_encoder = ContrastivePromptEncoder(**pe_params["params"])
        state_dict = torch.load(self.prompt_encoder_params["checkpoint"])
        contrastive_prompt_encoder.load_state_dict(state_dict)
        self.model.prompt_encoder.load_state_dict(
            contrastive_prompt_encoder.prompt_encoder.state_dict()
        )

    def init(self, params: dict):
        set_seed(params["train_params"]["seed"])
        self.seg_trainer = None
        logger.info("Parameters: ")
        write_yaml(params, file=sys.stdout)
        self.parse_params(params)

        kwargs = [
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ]
        logger.info("Creating Accelerator")
        self.accelerator = Accelerator(
            even_batches=False,
            kwargs_handlers=kwargs,
            split_batches=False,
            mixed_precision=self.train_params.get("precision", None),
        )
        logger.info("Initiliazing tracker...")
        self.plat_logger = get_experiment_logger(self.accelerator, self.params)
        self.url = self.plat_logger.url
        self.name = self.plat_logger.name
        self.train_loader, self.val_loaders, self.test_loader = get_dataloaders(
            self.dataset_params,
            self.dataloader_params,
            self.accelerator.num_processes,
        )
        model_name = self.model_params.get("name")
        logger.info(f"Creating model {model_name}")
        model_registry_params = deepcopy(self.model_params)
        model_registry_params.pop("name")
        self.model = model_registry[model_name](**model_registry_params)
        # load pretrained prompt encoder parameters
        self._load_prompt_encoder_parameters()

        self.watch_metric = self.train_params["watch_metric"]

        logger.info("Creating criterion")
        self.criterion = LabelAnythingLoss(**self.train_params["loss"])
        self.model = WrapperModule(self.model, self.criterion)
        self.input_image_size = self.model_params.get("image_size", SIZE)

        if self.train_params.get("compile", False):
            logger.info("Compiling model")
            self.model = torch.compile(self.model)
        logger.info("Preparing model, optimizer, dataloaders and scheduler")

        self.model = self.accelerator.prepare(self.model)

        if self.val_loaders:
            logger.info("Preparing validation dataloader")
            self._prep_for_validation()

        self._load_state()

    def _prep_for_training(self):
        logger.info("Creating optimizer")
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            params = self.model.module.get_learnable_params(self.train_params)
        else:
            params = self.model.get_learnable_params(self.train_params)
        self.optimizer = AdamW(
            params,
            lr=self.train_params["initial_lr"],
        )

        scheduler_params = self.train_params.get("scheduler", None)
        if scheduler_params:
            self.scheduler, self.scheduler_step_moment = get_scheduler(
                scheduler_params=scheduler_params,
                optimizer=self.optimizer,
                num_training_steps=self.train_params["max_epochs"]
                * len(self.train_loader),
            )

        self.train_loader, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.train_loader, self.optimizer, self.scheduler
        )

    def _prep_for_validation(self):
        self.val_loaders = {
            k: self.accelerator.prepare(v) for k, v in self.val_loaders.items()
        }

    def _load_state(self):
        if self.plat_logger.accelerator_state_dir:
            overwritten = False
            # Merge image_encoder dict with the state dict
            if (
                "checkpoint" in self.model_params
                and self.params["model"]["name"] != "lam_no_vit"
            ):
                if hasattr(self.model, "module"):
                    model = self.model.module.model
                else:
                    model = self.model.model
                shutil.copyfile(
                    self.plat_logger.accelerator_state_dir + "/pytorch_model.bin",
                    self.plat_logger.accelerator_state_dir + "/pytorch_model.bin.bak",
                )
                state_dict = torch.load(
                    self.plat_logger.accelerator_state_dir + "/pytorch_model.bin"
                )
                state_dict = {
                    **{
                        "model.image_encoder." + k: v
                        for k, v in model.image_encoder.state_dict().items()
                    },
                    **state_dict,
                }
                torch.save(
                    state_dict,
                    self.plat_logger.accelerator_state_dir + "/pytorch_model.bin",
                )
                overwritten = True

            try:
                self.accelerator.load_state(self.plat_logger.accelerator_state_dir)
                # Ripristinate old state
            finally:
                if (
                    "checkpoint" in self.model_params
                    and self.params["model"]["name"] != "lam_no_vit"
                    and overwritten
                ):
                    shutil.copyfile(
                        self.plat_logger.accelerator_state_dir
                        + "/pytorch_model.bin.bak",
                        self.plat_logger.accelerator_state_dir + "/pytorch_model.bin",
                    )
                    os.remove(
                        self.plat_logger.accelerator_state_dir
                        + "/pytorch_model.bin.bak"
                    )

    def launch(self):
        logger.info("Start training loop...")

        # Train the Model
        with self.plat_logger.train():
            logger.info(
                f"Running Model Training {self.params.get('experiment').get('name')}"
            )
            for epoch in range(self.train_params["max_epochs"]):
                logger.info(
                    "Epoch: {}/{}".format(epoch, self.train_params["max_epochs"])
                )
                self.train_epoch(epoch)

                metrics = None
                if self.val_loaders and epoch % self.train_params.get("val_frequency", 1) == 0:
                    with self.plat_logger.validate():
                        logger.info(f"Running Model Validation")
                        metrics = self.validate(epoch)
                        self._scheduler_step(SchedulerStepMoment.EPOCH, metrics)
                self.save_training_state(epoch, metrics)

        if self.test_loader:
            self.test()
        self.end()

    def save_training_state(self, epoch, metrics=None):
        if metrics:
            if self.best_metric is None:
                self.best_metric = metrics[self.watch_metric]
            if metrics[self.watch_metric] >= self.best_metric:
                logger.info(
                    f"Saving best model with metric {metrics[self.watch_metric]} as given that metric is greater than {self.best_metric}"
                )
                self.best_metric = metrics[self.watch_metric]
                self.plat_logger.log_training_state(epoch=epoch, subfolder="best")
        self.plat_logger.log_training_state(epoch=epoch, subfolder="latest")

    def _get_lr(self):
        if self.scheduler is None:
            return self.train_params["initial_lr"]
        try:
            if hasattr(self.scheduler, "get_lr"):
                return self.scheduler.get_lr()[0]
        except NotImplementedError:
            pass
        if hasattr(self.scheduler, "optimizer"):
            return self.scheduler.optimizer.param_groups[0]["lr"]
        return self.scheduler.optimizers[0].param_groups[0]["lr"]

    def _scheduler_step(self, moment, metrics=None):
        if moment != self.scheduler_step_moment or self.scheduler is None:
            return
        if moment == SchedulerStepMoment.BATCH:
            self.scheduler.step()
        elif moment == SchedulerStepMoment.EPOCH:
            self.scheduler.step(metrics[self.watch_metric])

    def _forward(
        self,
        batch_tuple: tuple[dict, torch.tensor],
        input_dict: dict,
        gt: torch.tensor,
        epoch: int,
        batch_idx: int,
    ):
        try:
            pass
            outputs = self.model(input_dict, gt)
        except RuntimeError as e:
            if "out of memory" in str(e):
                handle_oom(
                    self.model,
                    input_dict,
                    batch_tuple,
                    self.optimizer,
                    gt,
                    epoch,
                    batch_idx,
                )
                if self.oom:
                    raise e
                self.oom = True
                return e
            else:
                raise e
        self.oom = False
        return outputs

    def _backward(self, batch_idx, input_dict, outputs, gt, loss_normalizer):
        # loss_dict = compose_loss_input(input_dict, outputs)
        loss = outputs["loss"] / loss_normalizer
        self.accelerator.backward(loss)
        check_nan(
            self.model,
            input_dict,
            outputs,
            gt,
            loss,
            batch_idx,
            self.train_params,
        )
        return loss

    def _update_metrics(
        self,
        metrics: MetricCollection,
        preds: torch.tensor,
        gt: torch.tensor,
        tot_steps: int,
    ):
        with self.accelerator.no_sync(model=metrics):
            metrics_dict = metrics(preds, gt)
        if tot_steps % self.plat_logger.log_frequency == 0:
            for metric_name, metric_value in metrics_dict.items():
                metric_value = torch.mean(self.accelerator.gather(metric_value))
                self.plat_logger.log_metric(metric_name, metric_value)
        # .item() to all values
        metrics_dict = {k: v.item() for k, v in metrics_dict.items()}
        return metrics_dict

    def _update_val_metrics(
        self,
        metrics: MetricCollection,
        preds: torch.tensor,
        gt: torch.tensor,
        tot_steps,
    ):
        self.plat_logger.log_metric("step", self.global_val_step)
        return self._update_metrics(metrics, preds, gt, tot_steps)

    def _update_train_metrics(
        self,
        metrics: MetricCollection,
        previous_metric_values: dict,
        preds: torch.tensor,
        gt: torch,
        tot_steps: int,
        step: int,
    ):
        self.plat_logger.log_metric("step", self.global_train_step)
        if step == 0:
            metric_values = self._update_metrics(metrics, preds, gt, tot_steps)
        else:
            metric_values = previous_metric_values
        if tot_steps % self.plat_logger.log_frequency == 0:
            self.plat_logger.log_metric("lr", self._get_lr())
        return metric_values

    def train_epoch(
        self,
        epoch: int,
    ):
        if epoch > 0:
            set_seed(self.params["train_params"]["seed"] + epoch)
            logger.info(f"Setting seed to {self.params['train_params']['seed'] + epoch}")
        self.plat_logger.log_metric("start_epoch", epoch)
        self.model.train()
        accumulate_substitution = self.train_params.get(
            "accumulate_substitution", False
        )
        if accumulate_substitution and not self.train_params.get("substitute", True):
            raise ValueError(
                "accumulate_substitution can only be used when substitute is True"
            )

        # prepare metrics
        dataset_categories = next(
            iter(self.train_loader.dataset.datasets.values())
        ).categories
        num_classes = len(dataset_categories)
        metrics = MetricCollection(
            {
                "mIoU": DistributedMulticlassJaccardIndex(
                    num_classes=num_classes + 1,
                    average="macro",
                    ignore_index=-100,
                ),
                "FBIoU": DistributedBinaryJaccardIndex(
                    ignore_index=-100,
                ),
            },
            prefix="batch_",
        )
        metrics = self.accelerator.prepare(metrics)
        loss_avg = RunningAverage()

        # prepare substitutor
        substitutor = Substitutor(
            threshold=self.train_params.get("substitution_threshold", None),
            num_points=self.train_params.get("num_points", 1),
            substitute=self.train_params.get("substitute", True),
            long_side_length=self.dataset_params.get("common", {}).get(
                "image_size", SIZE
            ),
        )
        # allocate_memory(model, accelerator, optimizer, criterion, dataloader)

        # tqdm stuff
        bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            postfix={"loss": 0},
            desc=f"Train Epoch {epoch}/{self.train_params['max_epochs']-1}",
        )
        tot_steps = 0
        tot_images = 0
        loss_normalizer = 1
        self.oom = False
        metric_values = None

        # setting prompt encoder parameters
        if self.prompt_encoder_params:
            for p in self.model.module.model.prompt_encoder.parameters():
                p.requires_grad = epoch >= self.train_params.get(
                    "freeze_params_max_epoch", 0
                )

        for batch_idx, batch_tuple in bar:
            batch_tuple, dataset_names = batch_tuple
            cur_batch_size = get_batch_size(batch_tuple)
            loss_normalizer = (
                batch_tuple[1].shape[1] + 1
                if self.train_params.get("accumulate_substitution", True)
                else 1
            )
            substitutor.reset(batch=batch_tuple)
            for i, (input_dict, gt) in enumerate(substitutor):
                accumulating = accumulate_substitution and i != loss_normalizer - 1
                with nosync_accumulation(accumulating, self.accelerator, self.model):
                    result_dict = self._forward(
                        batch_tuple, input_dict, gt, epoch, batch_idx
                    )
                    if isinstance(result_dict, RuntimeError):
                        break
                    loss = self._backward(
                        batch_idx, input_dict, result_dict, gt, loss_normalizer
                    )
                    outputs = result_dict[ResultDict.LOGITS]
                    preds = outputs.argmax(dim=1)

                    if not accumulating:
                        self.optimizer.step()
                        self._scheduler_step(SchedulerStepMoment.BATCH)
                        self.optimizer.zero_grad()

                    loss_avg.update(loss.item())
                    self.plat_logger.log_metric("loss", loss.item())
                    glob_preds, glob_gt = to_global_multiclass(
                        input_dict["classes"], dataset_categories, preds, gt
                    )

                    metric_values = self._update_train_metrics(
                        metrics,
                        metric_values,
                        glob_preds,
                        glob_gt,
                        tot_steps,
                        i,
                    )
                    self.plat_logger.log_batch(
                        batch_idx=batch_idx,
                        image_idx=tot_images,
                        batch_size=cur_batch_size,
                        epoch=epoch,
                        step=tot_steps,
                        substitution_step=i,
                        input_dict=input_dict,
                        input_shape=self.input_image_size,
                        gt=gt,
                        pred=outputs,
                        dataset=self.train_loader.dataset,
                        dataset_names=dataset_names,
                        phase="train",
                        run_idx=0,  # Used for validation
                    )
                    substitutor.generate_new_points(outputs, gt)
                    bar.set_postfix(
                        {
                            **metric_values,
                            "loss": loss.item(),
                            "lr": self._get_lr(),
                        }
                    )
                    tot_steps += 1
                    self.global_train_step += 1
            self.plat_logger.save_experiment_timed()
            tot_images += cur_batch_size

        logger.info(f"Waiting for everyone")
        self.accelerator.wait_for_everyone()
        logger.info(f"Finished Epoch {epoch}")
        logger.info(f"Metrics")
        metric_dict = {
            **{
                "avg_" + k: v
                for k, v in {**metrics.compute(), **metrics.compute()}.items()
            },
            "avg_loss": loss_avg.compute(),
        }
        for k, v in metric_dict.items():
            logger.info(f"{k}: {v}")

        self.plat_logger.log_metrics(
            metrics=metric_dict,
            epoch=epoch,
        )

    def _add_to_json(self, input_dict, validation_run):
        if self.validation_json is None:
            return
        if validation_run is None:
            validation_run = 0
        if validation_run not in self.validation_json["json"]:
            self.validation_json["json"][validation_run] = []
        self.validation_json["json"][validation_run].append(
            input_dict[BatchKeys.IMAGE_IDS]
        )

    def validate(self, epoch: int, generate_json=False):
        metrics = []
        for name, dataloader in self.val_loaders.items():
            names = name.split("_")
            if len(names) > 1:
                name = names[-1]
            else:
                name = ""
            dataloader_metrics = self.validate_dataloader(
                name, dataloader, epoch, generate_json=generate_json
            )
            metrics.append(dataloader_metrics)
        mean_metrics = {
            k: torch.stack([torch.tensor(m[k]) for m in metrics]).mean()
            for k in metrics[0].keys()
        }
        logger.info(f"Validation epoch {epoch} finished")
        return mean_metrics

    def validate_dataloader(
        self, name, val_dataloader, epoch: int, generate_json=False
    ):
        logger.info(f"Validation of {name} epoch {epoch} started")
        if generate_json:
            self.validation_json = {
                "file": self.plat_logger.local_dir + f"/validation_image_ids.json",
                "json": {},
            }
        if self.train_params.get("validation_reruns", None) is None:
            metrics = self.validate_run(epoch, None)
        else:
            overall_metrics = []
            for validation_run in range(self.train_params["validation_reruns"]):
                metrics = self.validate_run(name, val_dataloader, epoch, validation_run)
                overall_metrics.append(metrics)
            metrics = {
                k: torch.stack([torch.tensor(m[k]) for m in overall_metrics]).mean()
                for k in overall_metrics[0].keys()
            }
            self.plat_logger.log_metrics(
                {**{"avg_" + k + name: v for k, v in metrics.items()}},
                epoch=epoch,
            )
            for k, v in metrics.items():
                logger.info(f"Validation epoch {epoch} - {name} - {k}: {v}")

        if generate_json:
            with open(self.validation_json["file"], "w") as f:
                json.dump(self.validation_json["json"], f)
            self.validation_json = None

        logger.info(f"Validation of {name} epoch {epoch} finished")
        return metrics

    def validate_run(self, name, val_loader, epoch, validation_run=None):
        if validation_run is None:
            seed = self.params["train_params"]["seed"]
            metrics_suffix = ""
        else:
            seed = self.params["train_params"]["seed"] + validation_run
            metrics_suffix = f"_{validation_run}"
        set_seed(seed)

        self.model.eval()
        avg_loss = RunningAverage()
        dataset_categories = next(iter(val_loader.dataset.datasets.values())).categories
        num_classes = len(dataset_categories)
        metrics = MetricCollection(
            {
                f"mIoU{metrics_suffix}": DistributedMulticlassJaccardIndex(
                    num_classes=num_classes + 1,
                    average="macro",
                    ignore_index=-100,
                ),
                f"FBIoU{metrics_suffix}": DistributedBinaryJaccardIndex(
                    ignore_index=-100,
                ),
            },
            prefix="batch_",
        )
        metrics = self.accelerator.prepare(metrics)

        tot_steps = 0
        tot_images = 0
        bar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            postfix={"loss": 0},
            desc=f"Validation Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process,
        )
        substitutor = Substitutor(substitute=False)

        with torch.no_grad():
            for batch_idx, batch_tuple in bar:
                batch_dict, dataset_names = batch_tuple
                substitutor.reset(batch=batch_dict)
                batch_dict = next(iter(substitutor))
                cur_batch_size = get_batch_size(batch_dict)
                image_dict, gt = batch_dict
                self._add_to_json(image_dict, validation_run)

                result_dict = self.model(image_dict, gt)
                outputs = result_dict[ResultDict.LOGITS]
                preds = outputs.argmax(dim=1)
                glob_preds, glob_gt = to_global_multiclass(
                    image_dict["classes"], dataset_categories, preds, gt
                )

                metrics_value = self._update_val_metrics(
                    metrics, glob_preds, glob_gt, tot_steps
                )
                loss = result_dict["loss"]

                avg_loss.update(loss.item())
                bar.set_postfix(
                    {
                        **metrics_value,
                        "loss": loss.item(),
                    }
                )
                self.plat_logger.log_batch(
                    batch_idx=batch_idx,
                    image_idx=tot_images,
                    batch_size=cur_batch_size,
                    epoch=epoch,
                    step=tot_steps,
                    substitution_step=0,
                    input_dict=image_dict,
                    input_shape=self.input_image_size,
                    gt=gt,
                    pred=outputs,
                    dataset=val_loader.dataset,
                    dataset_names=dataset_names,
                    phase="val",
                    run_idx=validation_run,
                )
                tot_steps += 1
                self.global_val_step += 1
                tot_images += cur_batch_size

            self.plat_logger.log_metrics(
                {
                    **{"avg_" + k + name: v for k, v in metrics.compute().items()},
                    "avg_loss": avg_loss.compute(),
                },
                epoch=epoch,
            )
        self.accelerator.wait_for_everyone()

        metrics_value = metrics.compute()
        for k, v in metrics_value.items():
            logger.info(
                f"Validation {metrics_suffix[1:]} - {name} - epoch {epoch} - {k}: {v}"
            )
        logger.info(
            f"Validation {metrics_suffix[1:]} epoch {epoch} - Loss: {avg_loss.compute()}"
        )
        return {
            "miou": metrics_value[f"batch_mIoU{metrics_suffix}"],
            "loss": avg_loss.compute(),
            "fbiou": metrics_value[f"batch_FBIoU{metrics_suffix}"],
        }

    def test(self):
        with self.plat_logger.test():
            for dataloader in self.test_loader:
                dataloader = self.accelerator.prepare(dataloader)
                self.test_dataset(dataloader=dataloader)

    def test_dataset(self, dataloader):
        self.model.eval()
        total_loss = 0
        metrics = MetricCollection(
            metrics=[
                self.accelerator.prepare(
                    DistributedMulticlassJaccardIndex(
                        # task="multiclass",
                        num_classes=dataloader.dataset.num_classes,
                        ignore_index=-100,
                    )
                ),
                self.accelerator.prepare(
                    DistributedBinaryJaccardIndex(ignore_index=-100)
                ),
            ]
        )
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.generate_class_embeddings = (
                self.model.module.generate_class_embeddings
            )
            self.model.predict = self.model.module.predict

        examples = dataloader.dataset.extract_prompts()
        self.model = set_class_embeddings(self.accelerator, self.model, examples)

        bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            postfix={"loss": 0},
            desc=f"Test: ",
            disable=not self.accelerator.is_local_main_process,
        )
        tot_images=0
        with torch.no_grad():
            for batch_idx, batch_dict in bar:
                image_dict, gt = batch_dict
                batch_size = image_dict["images"].size(0)
                outputs = self.model.predict(image_dict)
                # self.plat_logger.log_batch(
                #     batch_idx=batch_idx,
                #     image_idx=tot_images,
                #     batch_size=cur_batch_size,
                #     epoch=epoch,
                #     step=tot_steps,
                #     substitution_step=0,
                #     input_dict=image_dict,
                #     input_shape=self.input_image_size,
                #     gt=gt,
                #     pred=outputs,
                #     dataset=val_loader.dataset,
                #     dataset_names=dataset_names,
                #     phase="val",
                #     run_idx=validation_run,
                # )
                # total_loss += self.criterion(outputs, gt).item()  # sum up batch loss
                outputs = torch.argmax(outputs, dim=1)
                metrics.update(outputs, gt)
                tot_images += batch_size 
            # total_loss /= len(dataloader)
            metrics_values = metrics.compute()

            self.plat_logger.log_metrics(metrics=metrics_values)
            for k, v in metrics_values.items():
                logger.info(f"Test - {k}: {v}")
            # logger.info(f"Test - Loss: {total_loss}")

    def end(self):
        logger.info("Ending run")
        self.plat_logger.end()
        logger.info("Run ended")


class ParallelRun:
    slurm_command = "sbatch"
    slurm_script = "launch_run"
    slurm_script_first_parameter = "--parameters="
    slurm_outfolder = "out"
    out_extension = "out"
    param_extension = "yaml"
    slurm_stderr = "-e"
    slurm_stdout = "-o"

    def __init__(self, params: dict, experiment_timestamp: str):
        self.params = params
        self.exp_timestamp = experiment_timestamp
        if "." not in sys.path:
            sys.path.extend(".")

    def launch(self, only_create=False):
        subfolder = f"{self.exp_timestamp}_{self.params['experiment']['group']}"
        out_folder = os.path.join(self.slurm_outfolder, subfolder)
        os.makedirs(out_folder, exist_ok=True)

        run_uuid = str(uuid.uuid4())[:8]
        out_file = f"{run_uuid}.{self.out_extension}"
        out_file = os.path.join(out_folder, out_file)
        param_file = f"{run_uuid}.{self.param_extension}"
        param_file = os.path.join(out_folder, param_file)
        write_yaml(self.params, param_file)
        command = [
            self.slurm_command,
            self.slurm_stdout,
            out_file,
            self.slurm_stderr,
            out_file,
            self.slurm_script,
            self.slurm_script_first_parameter + param_file,
        ]
        if only_create:
            logger.info(f"Creating command: {' '.join(command)}")
        else:
            logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)
