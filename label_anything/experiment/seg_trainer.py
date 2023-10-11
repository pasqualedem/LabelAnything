import torch
from torch.cuda.amp import autocast
from torchmetrics import MetricCollection
import pandas as pd
import importlib
import tqdm

from logger.text_logger import get_logger
from logger.basesg_logger import BaseSGLogger as BaseLogger

from piptools.scripts.sync import _get_installed_distributions
from pprint import pformat
from typing import Mapping, Dict, Tuple
from label_anything.callbacks import callback_factory
from label_anything.callbacks.metrics import AuxMetricsUpdateCallback
from label_anything.models import MODELS as MODEL_DICT, WrappedModel
from utils.utils import get_module_class_from_path, instantiate_class


from super_gradients.common import MultiGPUMode
from super_gradients.common.data_types.enum import (
    MultiGPUMode,
    StrictLoad,
    EvaluationType,
)
from super_gradients.common.sg_loggers import SG_LOGGERS, BaseSGLogger
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger
from super_gradients.common.environment.device_utils import device_config
from super_gradients.training import utils as core_utils, models
from super_gradients.training import StrictLoad, Trainer
from super_gradients.training.params import TrainingParams
from super_gradients.training.utils import sg_trainer_utils
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
from super_gradients.training.utils.callbacks import (
    CallbackHandler,
    Phase,
    PhaseContext,
)

from super_gradients.training.metrics.metric_utils import (
    get_metrics_results_tuple,
    get_logging_values,
    get_train_loop_description_dict,
)
from super_gradients.training.utils.distributed_training_utils import (
    reduce_results_tuple_for_ddp,
    setup_device,
    get_gpu_mem_utilization,
    get_device_ids,
)


logger = get_logger(__name__)


class SegmentationTrainer(Trainer):
    def __init__(
        self,
        ckpt_root_dir=None,
        model_checkpoints_location: str = "local",
        num_devices=1,
        multi_gpu=MultiGPUMode.AUTO,
        **kwargs,
    ):
        self.run_id = None
        self.num_devices = num_devices
        self.multi_gpu = multi_gpu
        setup_device(num_gpus=num_devices, multi_gpu=multi_gpu)
        self.train_initialized = False
        self.validation = False
        self.model_checkpoints_location = model_checkpoints_location
        super().__init__(ckpt_root_dir=ckpt_root_dir, **kwargs)

    def init_dataset(self, dataset_interface, dataset_params):
        if isinstance(dataset_interface, str):
            try:
                module, dataset_interface_cls = get_module_class_from_path(
                    dataset_interface
                )
                dataset_module = importlib.import_module(module)
                dataset_interface_cls = getattr(dataset_module, dataset_interface_cls)
            except AttributeError as e:
                raise AttributeError("No interface found!") from e
            dataset_interface = dataset_interface_cls(dataset_params)
        elif isinstance(dataset_interface, type):
            pass
        else:
            raise ValueError("Dataset interface should be str or class")
        data_loader_num_workers = dataset_params.get("num_workers") or 0
        self.connect_dataset_interface(dataset_interface, data_loader_num_workers)
        return self.dataset_interface

    def connect_dataset_interface(self, dataset_interface, data_loader_num_workers):
        self.dataset_interface = dataset_interface
        (
            self.train_loader,
            self.valid_loader,
            self.test_loader,
            self.classes,
        ) = self.dataset_interface.get_data_loaders(
            batch_size_factor=self.num_devices,
            num_workers=data_loader_num_workers,
            distributed_sampler=self.multi_gpu
            == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL,
        )

        self.dataset_params = self.dataset_interface.get_dataset_params()

    def _add_metrics_update_callback(self, phase):
        """
        Adds AuxModelMetricsUpdateCallback to be fired at phase

        :param phase: Phase for the metrics callback to be fired at
        """
        self.phase_callbacks.append(AuxMetricsUpdateCallback(phase))

    def _load_model(self, params):
        model_params = params["model"]
        input_channels = len(params["dataset"]["channels"])
        output_channels = params["dataset"]["num_classes"]
        arch_params = {
            "input_channels": input_channels,
            "output_channels": output_channels,
            "in_channels": input_channels,
            "out_channels": output_channels,
            "num_classes": output_channels,
            **model_params["params"],
        }
        try:
            model = instantiate_class(model_params["name"], arch_params)
        except (AttributeError, ValueError):
            if model_params["name"] in MODELS_DICT.keys():
                model = MODELS_DICT[model_params["name"]](arch_params)
            else:
                model = models.get(
                    model_name=model_params["name"], arch_params=arch_params
                )

        return model, arch_params

    def init_model(self, params: Mapping, resume: bool, checkpoint_path: str = None):
        # load model
        model, arch_params = self._load_model(params)

        if "num_classes" not in arch_params.keys():
            if self.classes is None and self.dataset_interface is None:
                raise Exception(
                    "Error",
                    "Number of classes not defined in arch params and dataset is not defined",
                )
            else:
                arch_params["num_classes"] = len(self.classes)

        self.arch_params = core_utils.HpmStruct(**arch_params)
        self.net = model

        self._net_to_device()
        # SET THE FLAG FOR DIFFERENT PARAMETER GROUP OPTIMIZER UPDATE
        self.update_param_groups = hasattr(self.net.module, "update_param_groups")

        if resume and checkpoint_path is not None:
            self.checkpoint = load_checkpoint_to_model(
                ckpt_local_path=checkpoint_path,
                load_backbone=False,
                net=self.net,
                strict=StrictLoad.ON.value,
                load_weights_only=self.load_weights_only,
            )
            self.load_checkpoint = True

            if "ema_net" in self.checkpoint.keys():
                logger.warning(
                    "[WARNING] Main network has been loaded from checkpoint but EMA network exists as well. It "
                    " will only be loaded during validation when training with ema=True. "
                )

            # UPDATE TRAINING PARAMS IF THEY EXIST & WE ARE NOT LOADING AN EXTERNAL MODEL's WEIGHTS
            self.best_metric = (
                self.checkpoint["acc"] if "acc" in self.checkpoint.keys() else -1
            )
            self.start_epoch = (
                self.checkpoint["epoch"] if "epoch" in self.checkpoint.keys() else 0
            )

    def train(self, training_params: dict = dict()):
        super().train(
            model=self.net,
            training_params=training_params,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
        )
        if (
            self.train_loader.num_workers > 0
            and self.train_loader._iterator is not None
        ):
            self.train_loader._iterator._shutdown_workers()
        if (
            self.valid_loader.num_workers > 0
            and self.valid_loader._iterator is not None
        ):
            self.valid_loader._iterator._shutdown_workers()
        self._restore_best_params()

    def _train_epoch(self, epoch: int, silent_mode: bool = False) -> tuple:
        """
        train_epoch - A single epoch training procedure
            :param optimizer:   The optimizer for the network
            :param epoch:       The current epoch
            :param silent_mode: No verbosity
        """
        # SET THE MODEL IN training STATE
        self.net.train()
        # THE DISABLE FLAG CONTROLS WHETHER THE PROGRESS BAR IS SILENT OR PRINTS THE LOGS
        progress_bar_train_loader = tqdm(
            self.train_loader,
            bar_format="{l_bar}{bar:10}{r_bar}",
            dynamic_ncols=True,
            disable=silent_mode,
        )
        progress_bar_train_loader.set_description(f"Train epoch {epoch}")

        # RESET/INIT THE METRIC LOGGERS
        self._reset_metrics()

        self.train_metrics.to(device_config.device)
        loss_avg_meter = core_utils.utils.AverageMeter()

        context = PhaseContext(
            epoch=epoch,
            optimizer=self.optimizer,
            metrics_compute_fn=self.train_metrics,
            loss_avg_meter=loss_avg_meter,
            criterion=self.criterion,
            device=device_config.device,
            lr_warmup_epochs=self.training_params.lr_warmup_epochs,
            sg_logger=self.sg_logger,
            train_loader=self.train_loader,
            context_methods=self._get_context_methods(Phase.TRAIN_BATCH_END),
            ddp_silent_mode=self.ddp_silent_mode,
        )

        for batch_idx, batch_items in enumerate(progress_bar_train_loader):
            batch_items = core_utils.tensor_container_to_device(
                batch_items, device_config.device, non_blocking=True
            )
            (
                inputs,
                targets,
                additional_batch_items,
            ) = sg_trainer_utils.unpack_batch_items(batch_items)

            if self.pre_prediction_callback is not None:
                inputs, targets = self.pre_prediction_callback(
                    inputs, targets, batch_idx
                )

            context.update_context(
                batch_idx=batch_idx,
                inputs=inputs,
                target=targets,
                **additional_batch_items,
            )
            self.phase_callback_handler.on_train_batch_start(context)

            # AUTOCAST IS ENABLED ONLY IF self.training_params.mixed_precision - IF enabled=False AUTOCAST HAS NO EFFECT
            with autocast(enabled=self.training_params.mixed_precision):
                # FORWARD PASS TO GET NETWORK'S PREDICTIONS
                outputs = self._forward_step(inputs, targets, additional_batch_items)

                # COMPUTE THE LOSS FOR BACK PROP + EXTRA METRICS COMPUTED DURING THE LOSS FORWARD PASS
                loss, loss_log_items = self._get_losses(outputs, targets)

            context.update_context(preds=outputs, loss_log_items=loss_log_items)
            self.phase_callback_handler.on_train_batch_loss_end(context)

            # LOG LR THAT WILL BE USED IN CURRENT EPOCH AND AFTER FIRST WARMUP/LR_SCHEDULER UPDATE BEFORE WEIGHT UPDATE
            if not self.ddp_silent_mode and batch_idx == 0:
                self._write_lrs(epoch)

            self._backward_step(loss, epoch, batch_idx, context)

            # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
            logging_values = loss_avg_meter.average + get_metrics_results_tuple(
                self.train_metrics
            )
            gpu_memory_utilization = (
                get_gpu_mem_utilization() / 1e9 if torch.cuda.is_available() else 0
            )

            # RENDER METRICS PROGRESS
            pbar_message_dict = get_train_loop_description_dict(
                logging_values,
                self.train_metrics,
                self.loss_logging_items_names,
                gpu_mem=gpu_memory_utilization,
            )

            progress_bar_train_loader.set_postfix(**pbar_message_dict)
            self.phase_callback_handler.on_train_batch_end(context)

            # TODO: ITERATE BY MAX ITERS
            # FOR INFINITE SAMPLERS WE MUST BREAK WHEN REACHING LEN ITERATIONS.
            if (
                self._infinite_train_loader and batch_idx == len(self.train_loader) - 1
            ) or (
                self.max_train_batches is not None
                and self.max_train_batches - 1 <= batch_idx
            ):
                break

        self.train_monitored_values = sg_trainer_utils.update_monitored_values_dict(
            monitored_values_dict=self.train_monitored_values,
            new_values_dict=pbar_message_dict,
        )

        return logging_values

    def _forward_step(self, inputs, targets, additional_batch_items):
        if core_utils.get_param(self.training_params, "pass_targets_to_net", False):
            if core_utils.get_param(
                self.training_params, "pass_additional_batch_items_to_net", False
            ):
                return self.net(inputs, targets, additional_batch_items)
            return self.net(inputs, targets)
        if core_utils.get_param(
            self.training_params, "pass_additional_batch_items_to_net", False
        ):
            return self.net(inputs, additional_batch_items)
        return self.net(inputs)

    def _get_losses(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, tuple]:
        # GET THE OUTPUT OF THE LOSS FUNCTION
        if self.validation and core_utils.get_param(
            self.training_params, "inform_loss_in_validaiton", False
        ):
            loss = self.criterion(outputs, targets, validation=self.validation)
        else:
            loss = self.criterion(outputs, targets)
        if isinstance(loss, tuple):
            loss, loss_logging_items = loss
            # IF ITS NOT A TUPLE THE LOGGING ITEMS CONTAIN ONLY THE LOSS FOR BACKPROP (USER DEFINED LOSS RETURNS SCALAR)
        else:
            loss_logging_items = loss.unsqueeze(0).detach()

        # ON FIRST BACKWARD, DERRIVE THE LOGGING TITLES.
        if self.loss_logging_items_names is None or self._first_backward:
            self._init_loss_logging_names(loss_logging_items)
            if self.metric_to_watch:
                self._init_monitored_items()
            self._first_backward = False

        if len(loss_logging_items) != len(self.loss_logging_items_names):
            raise ValueError(
                "Loss output length must match loss_logging_items_names. Got "
                + str(len(loss_logging_items))
                + ", and "
                + str(len(self.loss_logging_items_names))
            )
        # RETURN AND THE LOSS LOGGING ITEMS COMPUTED DURING LOSS FORWARD PASS
        return loss, loss_logging_items

    def _validate_epoch(self, epoch: int, silent_mode: bool = False) -> tuple:
        self.validation = True
        res = super()._validate_epoch(epoch, silent_mode)
        self.validation = False
        return res

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        metrics: MetricCollection,
        evaluation_type: EvaluationType,
        epoch: int = None,
        silent_mode: bool = False,
        metrics_progress_verbose: bool = False,
    ):
        """
        Evaluates the model on given dataloader and metrics.

        :param data_loader: dataloader to perform evaluataion on
        :param metrics: (MetricCollection) metrics for evaluation
        :param evaluation_type: (EvaluationType) controls which phase callbacks will be used (for example, on batch end,
            when evaluation_type=EvaluationType.VALIDATION the Phase.VALIDATION_BATCH_END callbacks will be triggered)
        :param epoch: (int) epoch idx
        :param silent_mode: (bool) controls verbosity
        :param metrics_progress_verbose: (bool) controls the verbosity of metrics progress (default=False).
            Slows down the program significantly.

        :return: results tuple (tuple) containing the loss items and metric values.
        """

        # THE DISABLE FLAG CONTROLS WHETHER THE PROGRESS BAR IS SILENT OR PRINTS THE LOGS
        progress_bar_data_loader = tqdm(
            data_loader,
            bar_format="{l_bar}{bar:10}{r_bar}",
            dynamic_ncols=True,
            disable=silent_mode,
        )
        loss_avg_meter = core_utils.utils.AverageMeter()
        logging_values = None
        loss_tuple = None
        lr_warmup_epochs = (
            self.training_params.lr_warmup_epochs if self.training_params else None
        )
        context = PhaseContext(
            epoch=epoch,
            metrics_compute_fn=metrics,
            loss_avg_meter=loss_avg_meter,
            criterion=self.criterion,
            device=device_config.device,
            lr_warmup_epochs=lr_warmup_epochs,
            sg_logger=self.sg_logger,
            context_methods=self._get_context_methods(Phase.VALIDATION_BATCH_END),
        )

        if not silent_mode:
            # PRINT TITLES
            pbar_start_msg = (
                f"Validation epoch {epoch}"
                if evaluation_type == EvaluationType.VALIDATION
                else "Test"
            )
            progress_bar_data_loader.set_description(pbar_start_msg)

        with torch.no_grad():
            for batch_idx, batch_items in enumerate(progress_bar_data_loader):
                batch_items = core_utils.tensor_container_to_device(
                    batch_items, device_config.device, non_blocking=True
                )
                (
                    inputs,
                    targets,
                    additional_batch_items,
                ) = sg_trainer_utils.unpack_batch_items(batch_items)

                # TRIGGER PHASE CALLBACKS CORRESPONDING TO THE EVALUATION TYPE
                context.update_context(
                    batch_idx=batch_idx,
                    inputs=inputs,
                    target=targets,
                    **additional_batch_items,
                )
                if evaluation_type == EvaluationType.VALIDATION:
                    self.phase_callback_handler.on_validation_batch_start(context)
                else:
                    self.phase_callback_handler.on_test_batch_start(context)

                output = self._forward_step(inputs, targets, batch_items)
                context.update_context(preds=output)

                if self.criterion is not None:
                    # STORE THE loss_items ONLY, THE 1ST RETURNED VALUE IS THE loss FOR BACKPROP DURING TRAINING
                    loss_tuple = self._get_losses(output, targets)[1].cpu()
                    context.update_context(loss_log_items=loss_tuple)

                # TRIGGER PHASE CALLBACKS CORRESPONDING TO THE EVALUATION TYPE
                if evaluation_type == EvaluationType.VALIDATION:
                    self.phase_callback_handler.on_validation_batch_end(context)
                else:
                    self.phase_callback_handler.on_test_batch_end(context)

                # COMPUTE METRICS IF PROGRESS VERBOSITY IS SET
                if metrics_progress_verbose and not silent_mode:
                    # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
                    logging_values = get_logging_values(
                        loss_avg_meter, metrics, self.criterion
                    )
                    pbar_message_dict = get_train_loop_description_dict(
                        logging_values, metrics, self.loss_logging_items_names
                    )

                    progress_bar_data_loader.set_postfix(**pbar_message_dict)

                if (
                    evaluation_type == EvaluationType.VALIDATION
                    and self.max_valid_batches is not None
                    and self.max_valid_batches - 1 <= batch_idx
                ):
                    break

        # NEED TO COMPUTE METRICS FOR THE FIRST TIME IF PROGRESS VERBOSITY IS NOT SET
        if not metrics_progress_verbose:
            # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
            logging_values = get_logging_values(loss_avg_meter, metrics, self.criterion)
            pbar_message_dict = get_train_loop_description_dict(
                logging_values, metrics, self.loss_logging_items_names
            )

            progress_bar_data_loader.set_postfix(**pbar_message_dict)

        # TODO: SUPPORT PRINTING AP PER CLASS- SINCE THE METRICS ARE NOT HARD CODED ANYMORE (as done in
        #  calc_batch_prediction_accuracy_per_class in metric_utils.py), THIS IS ONLY RELEVANT WHEN CHOOSING
        #  DETECTIONMETRICS, WHICH ALREADY RETURN THE METRICS VALUEST HEMSELVES AND NOT THE ITEMS REQUIRED FOR SUCH
        #  COMPUTATION. ALSO REMOVE THE BELOW LINES BY IMPLEMENTING CRITERION AS A TORCHMETRIC.

        if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            logging_values = reduce_results_tuple_for_ddp(
                logging_values, next(self.net.parameters()).device
            )

        pbar_message_dict = get_train_loop_description_dict(
            logging_values, metrics, self.loss_logging_items_names
        )

        self.valid_monitored_values = sg_trainer_utils.update_monitored_values_dict(
            monitored_values_dict=self.valid_monitored_values,
            new_values_dict=pbar_message_dict,
        )

        if not silent_mode and evaluation_type == EvaluationType.VALIDATION:
            progress_bar_data_loader.write(
                "==========================================================="
            )
            sg_trainer_utils.display_epoch_summary(
                epoch=context.epoch,
                n_digits=4,
                train_monitored_values=self.train_monitored_values,
                valid_monitored_values=self.valid_monitored_values,
            )
            progress_bar_data_loader.write(
                "==========================================================="
            )

        return logging_values

    def _net_to_device(self):
        """
        Manipulates self.net according to device.multi_gpu
        """
        self.net.to(device_config.device)

        # FOR MULTI-GPU TRAINING (not distributed)
        sync_bn = core_utils.get_param(
            self.training_params, "sync_bn", default_val=False
        )
        if device_config.multi_gpu == MultiGPUMode.DATA_PARALLEL:
            self.net = torch.nn.DataParallel(self.net, device_ids=get_device_ids())
        elif device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            if sync_bn:
                if not self.ddp_silent_mode:
                    logger.info(
                        "DDP - Using Sync Batch Norm... Training time will be affected accordingly"
                    )
                self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net).to(
                    device_config.device
                )

            local_rank = int(device_config.device.split(":")[1])
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        elif not isinstance(self.net, WrappedModel):
            self.net = WrappedModel(self.net)
        else:
            pass

    def _restore_best_params(self):
        """
        Restore best parameters after the training
        """
        self.checkpoint = load_checkpoint_to_model(
            ckpt_local_path=self.checkpoints_dir_path + "/ckpt_best.pth",
            load_backbone=False,
            net=self.net,
            strict=StrictLoad.ON.value,
            load_weights_only=True,
        )

    def test(
        self,  # noqa: C901
        test_loader: torch.utils.data.DataLoader = None,
        loss: torch.nn.modules.loss._Loss = None,
        silent_mode: bool = False,
        test_metrics: Mapping = None,
        loss_logging_items_names=None,
        metrics_progress_verbose=False,
        test_phase_callbacks={},
        use_ema_net=True,
    ) -> dict:
        """
        Evaluates the model on given dataloader and metrics.

        :param test_loader: dataloader to perform test on.
        :param test_metrics: dict name: Metric for evaluation.
        :param silent_mode: (bool) controls verbosity
        :param metrics_progress_verbose: (bool) controls the verbosity of metrics progress (default=False). Slows down the program.
        :param use_ema_net (bool) whether to perform test on self.ema_model.ema (when self.ema_model.ema exists,
            otherwise self.net will be tested) (default=True)
        :return: results tuple (tuple) containing the loss items and metric values.

        All of the above args will override SgModel's corresponding attribute when not equal to None. Then evaluation
         is ran on self.test_loader with self.test_metrics.
        """
        test_loader = test_loader or self.test_loader
        loss = loss or self.training_params.loss
        self.validation = True
        if loss is not None:
            loss.to(self.device)
        test_phase_callbacks = [
            callback_factory(
                name,
                params,
                seg_trainer=self,
                dataset=self.dataset_interface,
                loader=test_loader,
            )
            for name, params in test_phase_callbacks.items()
        ]
        metrics_values = super().test(
            model=self.net,
            test_loader=test_loader,
            loss=loss,
            silent_mode=silent_mode,
            test_metrics_list=list(test_metrics.values()),
            loss_logging_items_names=loss_logging_items_names,
            metrics_progress_verbose=metrics_progress_verbose,
            test_phase_callbacks=test_phase_callbacks,
            use_ema_net=use_ema_net,
        )
        self.validation = False

        if self.test_loader.num_workers > 0:
            self.test_loader._iterator._shutdown_workers()

        metric_names = test_metrics.keys()
        loss_names = self.loss_logging_items_names
        if len(loss_names) + len(metric_names) != len(metrics_values):
            raise ValueError(
                f"Number of loss names ({len(loss_names)}) + number of metric names ({len(metric_names)}) "
                f"!= number of metrics values ({len(metrics_values)})"
            )
        losses = dict(zip(loss_names, metrics_values[: len(loss_names)]))
        metrics = dict(zip(metric_names, metrics_values[len(loss_names) :]))
        metrics = {**losses, **metrics}

        if "conf_mat" in metrics:
            metrics.pop("conf_mat")
            cf = test_metrics["conf_mat"].get_cf()
            logger.info(f"Confusion matrix:\n{cf}")
            self.sg_logger.add_table(
                "confusion_matrix",
                cf.cpu(),
                columns=list(self.dataset_interface.testset.id2label.values()),
                rows=list(self.dataset_interface.testset.id2label.values()),
            )
        self.sg_logger.add_summary(metrics)
        logger.info(f"Test metrics: {pformat(metrics)}")
        if "auc" in metrics:
            logger.info("Computing ROC curve...")
            roc = test_metrics["auc"].get_roc()
            fpr_tpr = [(roc[0][i], roc[1][i]) for i in range(len(roc))]
            skip = [x[0].shape.numel() // 1000 for x in fpr_tpr]
            fpr_tpr = [(fpr[::sk], tpr[::sk]) for (fpr, tpr), sk in zip(fpr_tpr, skip)]
            fprs, tprs = zip(*fpr_tpr)
            fprs = torch.cat(fprs)
            tprs = torch.cat(tprs)
            classes = list(self.dataset_interface.testset.id2label.values())
            cls = [[classes[i]] * len(fpr) for i, (fpr, tpr) in enumerate(fpr_tpr)]
            cls = [item for sublist in cls for item in sublist]
            df = pd.DataFrame({"class": cls, "fpr": fprs, "tpr": tprs})
            name = "ROCAUC"
            self.sg_logger.add_plot(name, df, "fpr", "tpr", classes_marker="class")
            logger.info("ROC curve computed.")
        return metrics

    def _init_monitored_items(self):
        if self.metric_to_watch == "loss":
            if len(self.loss_logging_items_names) == 1:
                self.metric_to_watch = self.loss_logging_items_names[0]
            else:
                raise ValueError(
                    f"Specify which loss {self.loss_logging_items_names} to watch"
                )
        return super()._init_monitored_items()

    def _initialize_sg_logger_objects(self, additional_configs_to_log: Dict = None):
        if not self.train_initialized:
            self.train_initialized = True
            # OVERRIDE SOME PARAMETERS TO MAKE SURE THEY MATCH THE TRAINING PARAMETERS
            general_sg_logger_params = {  # 'experiment_name': self.experiment_name,
                "experiment_name": "",
                "group": self.experiment_name,
                "storage_location": self.model_checkpoints_location,
                "resumed": self.load_checkpoint,
                "training_params": self.training_params,
                "checkpoints_dir_path": self.checkpoints_dir_path,
                "run_id": self.run_id,
            }
            sg_logger = core_utils.get_param(self.training_params, "sg_logger")

            if sg_logger is None:
                raise RuntimeError(
                    "logger must be defined in experiment params (see default_training_params)"
                )

            if isinstance(sg_logger, AbstractSGLogger):
                self.sg_logger = sg_logger
            elif isinstance(sg_logger, str):
                sg_logger_params = {
                    **general_sg_logger_params,
                    **core_utils.get_param(
                        self.training_params, "sg_logger_params", {}
                    ),
                }
                if sg_logger in SG_LOGGERS:
                    self.sg_logger = SG_LOGGERS[sg_logger](**sg_logger_params)
                elif sg_logger in LOGGERS:
                    self.sg_logger = LOGGERS[sg_logger](**sg_logger_params)
                else:
                    raise RuntimeError(
                        "sg_logger not defined in SG_LOGGERS of SuperGradients neither in LOGGERS of "
                        "EzDL"
                    )
            else:
                raise RuntimeError(
                    "sg_logger can be either an sg_logger name (str) or a subcalss of AbstractSGLogger"
                )

            if not (
                isinstance(self.sg_logger, BaseSGLogger)
                or isinstance(self.sg_logger, BaseLogger)
            ):
                logger.warning(
                    "WARNING! Using a user-defined sg_logger: files will not be automatically written to disk!\n"
                    "Please make sure the provided sg_logger writes to disk or compose your sg_logger to BaseSGLogger"
                )

            # IN CASE SG_LOGGER UPDATED THE DIR PATH
            self.checkpoints_dir_path = self.sg_logger.local_dir()
            self.training_params.override(sg_logger=sg_logger)
            additional_log_items = {
                "num_devices": self.num_devices,
                "multi_gpu": str(self.multi_gpu),
                "device_type": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu",
            }

            # ADD INSTALLED PACKAGE LIST + THEIR VERSIONS
            if self.training_params.log_installed_packages:
                pkg_list = list(
                    map(lambda pkg: str(pkg), _get_installed_distributions())
                )
                additional_log_items["installed_packages"] = pkg_list

            self.sg_logger.add_config("additional_log_items", additional_log_items)
            self.sg_logger.flush()

    def init_loggers(
        self,
        in_params: Mapping = None,
        train_params: Mapping = None,
        init_sg_loggers: bool = True,
        run_id=None,
    ) -> None:
        self.run_id = run_id
        if self.training_params is None:
            self.training_params = TrainingParams()
        self.training_params.override(**train_params)
        if init_sg_loggers:
            self._initialize_sg_logger_objects()
        if self.phase_callbacks is None:
            self.phase_callbacks = []
        self.phase_callback_handler = CallbackHandler(self.phase_callbacks)
        self.sg_logger.add_config(config=in_params)

    def _save_checkpoint(
        self,
        optimizer=None,
        epoch: int = None,
        validation_results_tuple: tuple = None,
        context: PhaseContext = None,
    ):
        """
        Save the current state dict as latest (always), best (if metric was improved), epoch# (if determined in training
        params)
        """
        # WHEN THE validation_results_tuple IS NONE WE SIMPLY SAVE THE state_dict AS LATEST AND Return
        if validation_results_tuple is None:
            self.sg_logger.add_checkpoint(
                tag="ckpt_latest_weights_only.pth",
                state_dict={"net": self.net.state_dict()},
                global_step=epoch,
            )
            return

        # COMPUTE THE CURRENT metric
        # IF idx IS A LIST - SUM ALL THE VALUES STORED IN THE LIST'S INDICES
        metric = (
            validation_results_tuple[self.metric_idx_in_results_tuple]
            if isinstance(self.metric_idx_in_results_tuple, int)
            else sum(
                [
                    validation_results_tuple[idx]
                    for idx in self.metric_idx_in_results_tuple
                ]
            )
        )

        # BUILD THE state_dict
        state = {"net": self.net.state_dict(), "acc": metric, "epoch": epoch}
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        if self.ema:
            state["ema_net"] = self.ema_model.ema.state_dict()
        # SAVES CURRENT MODEL AS ckpt_latest
        self.sg_logger.add_checkpoint(
            tag="ckpt_latest.pth", state_dict=state, global_step=epoch
        )

        # SAVE MODEL AT SPECIFIC EPOCHS DETERMINED BY save_ckpt_epoch_list
        if epoch in self.training_params.save_ckpt_epoch_list:
            self.sg_logger.add_checkpoint(
                tag=f"ckpt_epoch_{epoch}.pth", state_dict=state, global_step=epoch
            )

        # OVERRIDE THE BEST CHECKPOINT AND best_metric IF metric GOT BETTER THAN THE PREVIOUS BEST
        if (metric > self.best_metric and self.greater_metric_to_watch_is_better) or (
            metric < self.best_metric and not self.greater_metric_to_watch_is_better
        ):
            # STORE THE CURRENT metric AS BEST
            self.best_metric = metric
            self._save_best_checkpoint(epoch, state)

            # RUN PHASE CALLBACKS
            self.phase_callback_handler.on_validation_end_best_epoch(context)

            if isinstance(metric, torch.Tensor):
                metric = metric.item()
            logger.info(
                "Best checkpoint overriden: validation "
                + self.metric_to_watch
                + ": "
                + str(metric)
            )

        if self.training_params.average_best_models:
            net_for_averaging = self.ema_model.ema if self.ema else self.net
            state["net"] = self.model_weight_averaging.get_average_model(
                net_for_averaging, validation_results_tuple=validation_results_tuple
            )
            self.sg_logger.add_checkpoint(
                tag=self.average_model_checkpoint_filename,
                state_dict=state,
                global_step=epoch,
            )

    def _save_best_checkpoint(self, epoch, state):
        if self.ema:
            best_net = self.ema_model.ema
            state.pop("ema_net")
        else:
            best_net = self.net

        state["net"] = best_net.state_dict()
        self.sg_logger.add_checkpoint(
            tag=self.ckpt_best_name, state_dict=state, global_step=epoch
        )

    def run(
        self,
        data_loader: torch.utils.data.DataLoader,
        callbacks=None,
        silent_mode: bool = False,
    ):
        """
        Runs the model on given dataloader.

        :param data_loader: dataloader to perform run on

        """

        # THE DISABLE FLAG CONTROLS WHETHER THE PROGRESS BAR IS SILENT OR PRINTS THE LOGS
        if callbacks is None:
            callbacks = []
        progress_bar_data_loader = tqdm(
            data_loader,
            bar_format="{l_bar}{bar:10}{r_bar}",
            dynamic_ncols=True,
            disable=silent_mode,
        )
        context = PhaseContext(
            criterion=self.criterion, device=self.device, sg_logger=self.sg_logger
        )

        self.phase_callbacks.extend(callbacks)

        if not silent_mode:
            # PRINT TITLES
            pbar_start_msg = f"Running model on {len(data_loader)} batches"
            progress_bar_data_loader.set_description(pbar_start_msg)

        with torch.no_grad():
            for batch_idx, batch_items in enumerate(progress_bar_data_loader):
                batch_items = core_utils.tensor_container_to_device(
                    batch_items, self.device, non_blocking=True
                )

                additional_batch_items = {}
                targets = None
                if hasattr(batch_items, "__len__"):
                    if len(batch_items) == 2:
                        inputs, targets = batch_items
                    elif len(batch_items) == 3:
                        inputs, targets, additional_batch_items = batch_items
                    else:
                        raise ValueError(
                            f"Expected 1, 2 or 3 items in batch_items, got {len(batch_items)}"
                        )
                else:
                    inputs = batch_items

                output = self.net(inputs)

                context.update_context(
                    batch_idx=batch_idx,
                    inputs=inputs,
                    preds=output,
                    target=targets,
                    **additional_batch_items,
                )

                # TRIGGER PHASE CALLBACKS
                self.phase_callback_handler(Phase.POST_TRAINING, context)
