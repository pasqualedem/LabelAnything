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
from transformers import get_scheduler

from label_anything.data import get_dataloaders
from label_anything.experiment.substitution import Substitutor
from label_anything.logger.image_logger import Logger
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
from label_anything.utils.utils import RunningAverage, write_yaml

from .utils import (
    allocate_memory,
    check_nan,
    get_batch_size,
    handle_oom,
    set_class_embeddings,
    parse_params,
    comet_experiment,
)

logger = get_logger(__name__)


class Run:
    def __init__(self):
        self.kd = None
        self.params = None
        self.dataset = None
        self.experiment = None
        self.comet_logger = None
        self.dataset_params = None
        self.train_params = None
        self.model = None
        self.scheduler = None
        self.criterion = None
        self.oom = None
        if "." not in sys.path:
            sys.path.extend(".")

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
        self.parse_params(params)
        (
            self.train_params,
            self.dataset_params,
            self.dataloader_params,
            self.model_params,
        ) = parse_params(params)

        comet_params = self.params.get("logger", {}).get("comet", {})
        comet_information = {
            "api_key": os.getenv("COMET_API_KEY"),
            "project_name": self.params["experiment"]["name"],
            **comet_params,
        }
        kwargs = [
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ]
        self.accelerator = Accelerator(
            even_batches=False, kwargs_handlers=kwargs, split_batches=False
        )
        self.comet_logger = comet_experiment(
            comet_information, self.accelerator, self.params
        )
        self.url = self.comet_logger.experiment.url
        self.name = self.comet_logger.experiment.name

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            self.dataset_params, self.dataloader_params, self.accelerator.num_processes
        )
        model_name = self.model_params.pop("name")
        self.model = model_registry[model_name](**self.model_params)

    def launch(self):
        logger.info("Start training loop...")

        self.criterion = LabelAnythingLoss(**self.train_params["loss"])
        self.optimizer = AdamW(
            self.model.get_learnable_params(self.train_params),
            lr=self.train_params["initial_lr"],
        )

        scheduler_params = self.train_params.get("scheduler", None)
        if scheduler_params:
            self.scheduler = get_scheduler(
                scheduler_params["type"],
                optimizer=self.optimizer,
                num_warmup_steps=scheduler_params["warmup_steps"],
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

        # Train the Model
        with self.comet_logger.experiment.train():
            logger.info(
                f"Running Model Training {self.params.get('experiment').get('name')}"
            )
            for epoch in range(self.train_params["max_epochs"]):
                logger.info(
                    "Epoch: {}/{}".format(epoch, self.train_params["max_epochs"])
                )
                self.train_epoch(epoch)

                self.comet_logger.log_training_state(epoch=epoch)
                if self.val_loader:
                    with self.comet_logger.experiment.validate():
                        logger.info(f"Running Model Validation")
                        self.validate(epoch)
                        self.comet_logger.save_experiment()

        if self.test_loader:
            with self.comet_logger.experiment.test():
                for dataloader, support_dataset in self.test_loader:
                    dataloader = self.accelerator.prepare(dataloader)
                    self.test(
                        dataloader=dataloader,
                        train_dataset=support_dataset,
                    )

        logger.info("Ending run")
        self.comet_logger.end()
        logger.info("Run ended")

    def _get_lr(self):
        return (
            self.scheduler.get_lr()[0]
            if self.scheduler
            else self.train_params["initial_lr"]
        )

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
        if tot_steps % self.comet_logger.log_frequency == 0:
            for metric_name, metric_value in metrics_dict.items():
                self.comet_logger.log_metric(metric_name, metric_value)
        return metrics_dict

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
        if tot_steps % self.comet_logger.log_frequency == 0:
            metric_values = self._update_metrics(metrics, preds, gt, outputs, tot_steps)
            if i == 0:
                first_step_loss_avg.update(loss.item())
                self.comet_logger.log_metric("first_step_loss", loss.item())
                if tot_steps % self.comet_logger.log_frequency == 0:
                    self._update_metrics(
                        first_step_metrics, preds, gt, outputs, tot_steps
                    )
                    self.comet_logger.log_metric("lr", self._get_lr())
        else:
            metric_values = previous_metric_values
        return metric_values

    def train_epoch(
        self,
        epoch: int,
    ):
        self.model.train()
        accumulate_substitution = self.train_params.get(
            "accumulate_substitution", False
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
                outputs = self._forward(batch_tuple, input_dict, gt, epoch, batch_idx)
                if isinstance(outputs, RuntimeError):
                    break
                loss = self._backward(
                    batch_idx, input_dict, outputs, gt, loss_normalizer
                )
                preds = outputs.argmax(dim=1)

                if not accumulate_substitution or i == loss_normalizer - 1:
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                loss_avg.update(loss.item())
                self.comet_logger.log_metric("loss", loss.item())

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
                self.comet_logger.log_batch(
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
            self.comet_logger.save_experiment_timed()
            tot_images += cur_batch_size

        logger.info(f"Waiting for everyone")
        self.accelerator.wait_for_everyone()
        logger.info(f"Finished Epoch {epoch}")
        logger.info(f"Metrics")
        metric_dict = {
            **first_step_metrics.compute(),
            **metrics.compute(),
            "loss": loss_avg.compute(),
            "first_step_loss": first_step_loss_avg.compute(),
        }
        for k, v in metric_dict.items():
            logger.info(f"{k}: {v}")

        self.comet_logger.log_metrics(
            metrics=metric_dict,
            epoch=epoch,
        )

    def validate(self, epoch):
        self.val_loader.dataset.reset_seed(42)
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

                metrics_value = self._update_metrics(
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
                self.comet_logger.log_batch(
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
                tot_images += cur_batch_size

            self.comet_logger.log_metrics(
                {
                    **metrics.compute(),
                    "loss": avg_loss.compute(),
                },
                epoch=epoch,
            )
        self.accelerator.wait_for_everyone()
        logger.info(f"Validation epoch {epoch} finished")
        for k, v in metrics.compute().items():
            logger.info(f"Validation epoch {epoch} - {k}: {v}")
        logger.info(f"Validation epoch {epoch} - Loss: {avg_loss.compute()}")

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

            self.comet_logger.log_metrics(metrics=metrics_values)
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
