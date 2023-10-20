import torch
import pandas as pd

from collections import defaultdict
from super_gradients.training.utils.callbacks import (
    PhaseCallback,
    PhaseContext,
    Phase,
    Callback,
    MetricsUpdateCallback,
)
from super_gradients.training.utils.utils import AverageMeter
from super_gradients.training.models.kd_modules.kd_module import KDOutput


# Deprecated
class MetricsLogCallback(PhaseCallback):
    """
    A callback that logs metrics to MLFlow.
    """

    def __init__(self, phase: Phase, freq: int):
        """
        param phase: phase to log metrics for
        param freq: frequency of logging
        param client: MLFlow client
        """

        if phase == Phase.TRAIN_EPOCH_END:
            self.prefix = "train_"
        elif phase == Phase.VALIDATION_EPOCH_END:
            self.prefix = "val_"
        else:
            raise NotImplementedError("Unrecognized Phase")

        super(MetricsLogCallback, self).__init__(phase)
        self.freq = freq

    def __call__(self, context: PhaseContext):
        """
        Logs metrics to MLFlow.
            param context: context of the current phase
        """
        if self.phase == Phase.TRAIN_EPOCH_END:
            context.sg_logger.add_summary({"epoch": context.epoch})
        if context.epoch % self.freq == 0:
            context.sg_logger.add_scalars(
                {self.prefix + k: v for k, v in context.metrics_dict.items()}
            )


class AverageMeterCallback(PhaseCallback):
    def __init__(self):
        super(AverageMeterCallback, self).__init__(Phase.TEST_BATCH_END)
        self.meters = {}

    def __call__(self, context: PhaseContext):
        """
        Logs metrics to MLFlow.
            param context: context of the current phase
        """
        context.metrics_compute_fn.update(context.preds, context.target)
        metrics_dict = context.metrics_compute_fn.compute()
        for k, v in metrics_dict.items():
            if not self.meters.get(k):
                self.meters[k] = AverageMeter()
            self.meters[k].update(v, 1)


class AuxMetricsUpdateCallback(MetricsUpdateCallback):
    """
    A callback that updates metrics for the current phase for a model with auxiliary outputs.
    """

    def __init__(self, phase: Phase):
        super().__init__(phase=phase)

    def __call__(self, context: PhaseContext):
        def is_composed_output(x):
            return isinstance(x, ComposedOutput) or isinstance(x, KDOutput)

        def get_output(x):
            return (
                x.main
                if isinstance(x, ComposedOutput)
                else x.student_output
                if isinstance(x, KDOutput)
                else x
            )

        metrics_compute_fn_kwargs = {
            k: get_output(v) if k == "preds" and is_composed_output(v) else v
            for k, v in context.__dict__.items()
        }
        context.metrics_compute_fn.update(**metrics_compute_fn_kwargs)
        if context.criterion is not None:
            context.loss_avg_meter.update(context.loss_log_items, len(context.inputs))


class PerExampleMetricCallback(Callback):
    def __init__(self, phase: Phase):
        super().__init__()
        self.metrics_dict = defaultdict(dict)

    def register(self, img_name, metric_name, res):
        if isinstance(res, dict):
            for sub_name, value in res.items():
                self.register(img_name, sub_name, value)
        elif isinstance(res, torch.Tensor):
            self.metrics_dict[metric_name][img_name] = res.item()
        else:
            self.metrics_dict[metric_name][img_name] = res

    def on_test_batch_end(self, context: PhaseContext):
        metrics = context.metrics_compute_fn.clone()
        preds = context.preds
        if isinstance(context.preds, KDOutput):
            preds = preds.student_output
        if isinstance(context.preds, ComposedOutput):
            preds = preds.main
        for pred, gt, name, padding in zip(
            preds, context.target, context.input_name, context.padding
        ):
            metrics.reset()
            for metric_name, metric_fn in metrics.items():
                res = metric_fn(pred.unsqueeze(0), gt.unsqueeze(0), [padding])
                self.register(name, metric_name, res)

    def on_test_loader_end(self, context: PhaseContext) -> None:
        test_results = pd.DataFrame(self.metrics_dict)
        context.sg_logger.add_table(
            "test_results", test_results, test_results.shape[1], test_results.shape[0]
        )
