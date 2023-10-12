from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import *

from .segmentation import SegmentationVisualizationCallback
from .detection import DetectionVisualizationCallback


def callback_factory(name, params, **kwargs):
    """
    Creates a callback from a name and parameters.
    Args:
        name: name of the callback
        params: parameters for the callback
    Returns:
        initializated callback
    """
    params = params or {}
    if not isinstance(name, str):  # Assuming already a callback
        return name
    if name in ["early_stop", "early_stopping", "EarlyStop"]:
        if "phase" in params:
            params.pop("phase")
        return EarlyStop(Phase.VALIDATION_EPOCH_END, **params)
    if name == "SegmentationVisualizationCallback":
        seg_trainer = kwargs["seg_trainer"]
        loader = kwargs["loader"]
        dataset = kwargs["dataset"]
        params["freq"] = params.get("freq", 1)
        params["phase"] = (
            Phase.VALIDATION_BATCH_END
            if params["phase"] == "validation"
            else Phase.TEST_BATCH_END
        )
        if params["phase"] == Phase.TEST_BATCH_END:
            params["batch_idxs"] = range(len(loader))
        if hasattr(dataset, "cmap"):
            params["cmap"] = dataset.cmap
        return SegmentationVisualizationCallback(
            logger=seg_trainer.sg_logger,
            last_img_idx_in_batch=4,
            id2label=dataset.trainset.id2label,
            undo_preprocessing=dataset.undo_preprocess,
            **params
        )
    if name == "DetectionVisualizationCallback":
        seg_trainer = kwargs["seg_trainer"]
        loader = kwargs["loader"]
        dataset = kwargs["dataset"]
        params["freq"] = params.get("freq", 1)
        params["phase"] = {
            "validation": Phase.VALIDATION_BATCH_END,
            "train": Phase.TRAIN_BATCH_END,
            "test": Phase.TEST_BATCH_END,
        }[params["phase"]]
        if params["phase"] == Phase.TEST_BATCH_END:
            params["batch_idxs"] = range(len(loader))
        return DetectionVisualizationCallback(
            logger=seg_trainer.sg_logger,
            last_img_idx_in_batch=4,
            id2label=dataset.trainset.id2label,
            undo_preprocessing=dataset.undo_preprocess,
            **params
        )
    if params.get("phase"):
        params["phase"] = Phase.__dict__[params.get("phase")]
    return globals()[name](**params)
