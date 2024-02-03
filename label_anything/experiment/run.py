import copy
import os
import random
import subprocess
import sys
import tempfile
import uuid
from copy import deepcopy

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from tqdm import tqdm

from label_anything.data import get_dataloaders
from label_anything.experiment.substitution import Substitutor
from label_anything.logger.text_logger import get_logger
from label_anything.loss import LabelAnythingLoss
from label_anything.models import model_registry
from label_anything.utils.metrics import (
    AverageMetricCollection,
    MetricCollection,
    FBIoU,
    JaccardIndex,
    fbiou,
    multiclass_jaccard_index,
)
from label_anything.utils.utils import FLOAT_PRECISIONS, RunningAverage, write_yaml

from .utils import (
    SchedulerStepMoment,
    allocate_memory,
    check_nan,
    get_batch_size,
    handle_oom,
    set_class_embeddings,
    parse_params,
    get_experiment_logger,
    nosync_accumulation,
    get_scheduler,
)

logger = get_logger(__name__)


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

    def parse_params(self, params: dict):
        self.params = deepcopy(params)

        (
            self.train_params,
            self.dataset_params,
            self.dataloader_params,
            self.model_params,
        ) = parse_params(self.params)

    def init(self, params: dict):
        # set torch, numpy, random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        self.seg_trainer = None
        logger.info("Parameters: ")
        logger.info(params)
        self.parse_params(params)
        (
            self.train_params,
            self.dataset_params,
            self.dataloader_params,
            self.model_params,
        ) = parse_params(params)

        kwargs = [
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ]
        self.accelerator = Accelerator(
            even_batches=False,
            kwargs_handlers=kwargs,
            split_batches=False,
            mixed_precision=self.train_params.get("precision", None),
        )
        self.plat_logger = get_experiment_logger(self.accelerator, self.params)
        self.url = self.plat_logger.url
        self.name = self.plat_logger.name
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            self.dataset_params,
            self.dataloader_params,
            self.accelerator.num_processes,
        )
        model_name = self.model_params.pop("name")
        self.model = model_registry[model_name](**self.model_params)

        self.watch_metric = self.train_params["watch_metric"]

    def launch(self):
        logger.info("Start training loop...")

        self.criterion = LabelAnythingLoss(**self.train_params["loss"])
        self.optimizer = AdamW(
            self.model.get_learnable_params(self.train_params),
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

        if self.train_params.get("compile", False):
            self.model = torch.compile(self.model)

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        if self.val_loader:
            self.val_loader = self.accelerator.prepare(self.val_loader)
            
        if self.plat_logger.accelerator_state_dir:
            self.accelerator.load_state(self.plat_logger.accelerator_state_dir)

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
                if self.val_loader:
                    with self.plat_logger.validate():
                        logger.info(f"Running Model Validation")
                        metrics = self.validate(epoch)
                        self._scheduler_step(SchedulerStepMoment.EPOCH, metrics)
                self.save_training_state(epoch, metrics)

        if self.test_loader:
            with self.plat_logger.test():
                for dataloader, support_dataset in self.test_loader:
                    dataloader = self.accelerator.prepare(dataloader)
                    self.test(
                        dataloader=dataloader,
                        train_dataset=support_dataset,
                    )

        logger.info("Ending run")
        self.plat_logger.end()
        logger.info("Run ended")

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
        if hasattr(self.scheduler, "get_lr"):
            return self.scheduler.get_lr()[0]
        return self.scheduler.optimizer.param_groups[0]["lr"]

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
            outputs = self.model(input_dict)
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
        loss = self.criterion(outputs, gt) / loss_normalizer
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
        metrics: AverageMetricCollection,
        preds: torch.tensor,
        gt: torch.tensor,
        outputs,
        tot_steps,
    ):
        metrics_dict = metrics.update(
            preds, gt, num_classes=outputs.shape[1], ignore_index=-100
        )
        if tot_steps % self.plat_logger.log_frequency == 0:
            for metric_name, metric_value in metrics_dict.items():
                self.plat_logger.log_metric(metric_name, metric_value)
        return metrics_dict

    def _update_val_metrics(
        self,
        metrics: AverageMetricCollection,
        preds: torch.tensor,
        gt: torch.tensor,
        outputs,
        tot_steps,
    ):
        self.plat_logger.log_metric("step", self.global_val_step)
        return self._update_metrics(metrics, preds, gt, outputs, tot_steps)

    def _update_train_metrics(
        self,
        metrics: AverageMetricCollection,
        first_step_metrics: AverageMetricCollection,
        previous_metric_values: dict,
        preds: torch.tensor,
        gt: torch,
        outputs: torch.tensor,
        tot_steps: int,
        i: int,
        loss: torch.tensor,
        first_step_loss_avg: RunningAverage,
    ):
        self.plat_logger.log_metric("step", self.global_train_step)
        if tot_steps % self.plat_logger.log_frequency == 0:
            metric_values = self._update_metrics(metrics, preds, gt, outputs, tot_steps)
            if i == 0:
                first_step_loss_avg.update(loss.item())
                self.plat_logger.log_metric("first_step_loss", loss.item())
                if tot_steps % self.plat_logger.log_frequency == 0:
                    self._update_metrics(
                        first_step_metrics, preds, gt, outputs, tot_steps
                    )
                    self.plat_logger.log_metric("lr", self._get_lr())
        else:
            metric_values = previous_metric_values
        return metric_values

    def train_epoch(
        self,
        epoch: int,
    ):
        self.plat_logger.log_metric("start_epoch", epoch)
        self.model.train()
        accumulate_substitution = self.train_params.get(
            "accumulate_substitution", False
        )
        if accumulate_substitution and not self.train_params.get("substitute", True):
            raise ValueError(
                "accumulate_substitution can only be used when substitute is True"
            )
        metrics = AverageMetricCollection(
            prefix="batch_",
            metrics={
                "mIoU": multiclass_jaccard_index,
                "FBIoU": fbiou,
            },
            accelerator=self.accelerator,
        )
        first_step_metrics = AverageMetricCollection(
            prefix="first_step_",
            metrics={
                "mIoU": multiclass_jaccard_index,
                "FBIoU": fbiou,
            },
            accelerator=self.accelerator,
        )
        substitutor = Substitutor(
            threshold=self.train_params.get("substitution_threshold", None),
            num_points=self.train_params.get("num_points", 1),
            substitute=self.train_params.get("substitute", True),
        )
        loss_avg = RunningAverage()
        first_step_loss_avg = RunningAverage()
        # allocate_memory(model, accelerator, optimizer, criterion, dataloader)

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
                    outputs = self._forward(
                        batch_tuple, input_dict, gt, epoch, batch_idx
                    )
                    if isinstance(outputs, RuntimeError):
                        break
                    loss = self._backward(
                        batch_idx, input_dict, outputs, gt, loss_normalizer
                    )
                    preds = outputs.argmax(dim=1)

                    if not accumulating:
                        self.optimizer.step()
                        self._scheduler_step(SchedulerStepMoment.BATCH)
                        self.optimizer.zero_grad()

                    loss_avg.update(loss.item())
                    self.plat_logger.log_metric("loss", loss.item())

                    metric_values = self._update_train_metrics(
                        metrics,
                        first_step_metrics,
                        metric_values,
                        preds,
                        gt,
                        outputs,
                        tot_steps,
                        i,
                        loss,
                        first_step_loss_avg,
                    )
                    self.plat_logger.log_batch(
                        batch_idx=batch_idx,
                        image_idx=tot_images,
                        batch_size=cur_batch_size,
                        epoch=epoch,
                        step=tot_steps,
                        substitution_step=i,
                        input_dict=input_dict,
                        gt=gt,
                        pred=outputs,
                        dataset=self.train_loader.dataset,
                        dataset_names=dataset_names,
                        phase="train",
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
            "avg_first_step_loss": first_step_loss_avg.compute(),
        }
        for k, v in metric_dict.items():
            logger.info(f"{k}: {v}")

        self.plat_logger.log_metrics(
            metrics=metric_dict,
            epoch=epoch,
        )

    def validate(self, epoch):
        self.val_loader.dataset.reset_seed(self.params["train_params"]["seed"])
        self.model.eval()
        avg_loss = RunningAverage()
        metrics = AverageMetricCollection(
            prefix="batch_",
            metrics={
                "mIoU": multiclass_jaccard_index,
                "FBIoU": fbiou,
            },
            accelerator=self.accelerator,
        )

        tot_steps = 0
        tot_images = 0
        bar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
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

                outputs = self.model(image_dict)
                preds = outputs.argmax(dim=1)

                metrics_value = self._update_val_metrics(
                    metrics, preds, gt, outputs, tot_steps
                )
                loss = torch.mean(self.accelerator.gather(self.criterion(outputs, gt)))

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
                    gt=gt,
                    pred=outputs,
                    dataset=self.val_loader.dataset,
                    dataset_names=dataset_names,
                    phase="val",
                )
                tot_steps += 1
                self.global_val_step += 1
                tot_images += cur_batch_size

            self.plat_logger.log_metrics(
                {
                    **{"avg_" + k: v for k, v in metrics.compute().items()},
                    "avg_loss": avg_loss.compute(),
                },
                epoch=epoch,
            )
        self.accelerator.wait_for_everyone()
        logger.info(f"Validation epoch {epoch} finished")
        metrics_value = metrics.compute()
        for k, v in metrics_value.items():
            logger.info(f"Validation epoch {epoch} - {k}: {v}")
        logger.info(f"Validation epoch {epoch} - Loss: {avg_loss.compute()}")
        return {"miou": metrics_value["batch_mIoU"], "loss": avg_loss.compute()}

    def test(self, dataloader, train_dataset):
        self.model.eval()
        total_loss = 0
        metrics = MetricCollection(
            metrics=[
                self.accelerator.prepare(
                    JaccardIndex(
                        task="multiclass",
                        num_classes=dataloader.dataset.num_classes,
                        ignore_index=-100,
                    )
                ),
                self.accelerator.prepare(FBIoU(ignore_index=-100)),
            ]
        )
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.generate_class_embeddings = (
                self.model.module.generate_class_embeddings
            )
            self.model.predict = self.model.module.predict

        examples = dataloader.dataset.extract_prompts(
            train_dataset.cat2img,
            train_dataset.img2cat,
            train_dataset.images,
            train_dataset.img2cat_annotations,
        )
        self.model = set_class_embeddings(self.model, examples)

        bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            postfix={"loss": 0},
            desc=f"Test: ",
            disable=not self.accelerator.is_local_main_process,
        )

        with torch.no_grad():
            for batch_idx, batch_dict in bar:
                image_dict, gt = batch_dict

                output = self.model.predict(image_dict)
                total_loss += self.criterion(output, gt).item()  # sum up batch loss
                output = torch.argmax(output, dim=1)
                metrics.update(output, gt)

            total_loss /= len(dataloader)
            metrics_values = metrics.compute()

            self.plat_logger.log_metrics(metrics=metrics_values)
            for k, v in metrics_values.items():
                logger.info(f"Test - {k}: {v}")
            logger.info(f"Test - Loss: {total_loss}")


class ParallelRun:
    slurm_command = "sbatch"
    slurm_script = "launch_run"
    slurm_script_first_parameter = "--parameters="
    slurm_output = "out/run"
    out_extension = "out"
    slurm_stderr = "-e"
    slurm_stdout = "-o"

    def __init__(self, params: dict, experiment_uuid: str):
        self.params = params
        self.exp_uuid = experiment_uuid
        if "." not in sys.path:
            sys.path.extend(".")

    def launch(self, only_create=False):
        os.makedirs("out", exist_ok=True)
        tmp_parameters_file = tempfile.NamedTemporaryFile(delete=False)
        write_yaml(self.params, tmp_parameters_file.name)
        tmp_parameters_file.close()
        out_file = f"{self.slurm_output}_{self.exp_uuid}_{str(uuid.uuid4())[:8]}.{self.out_extension}"
        command = [
            self.slurm_command,
            self.slurm_stdout,
            out_file,
            self.slurm_stderr,
            out_file,
            self.slurm_script,
            self.slurm_script_first_parameter + tmp_parameters_file.name,
        ]
        if only_create:
            logger.info(f"Creating command: {' '.join(command)}")
        else:
            logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)
