from typing import Mapping, Union, Callable, List

import torch
import numpy as np

from super_gradients.training.utils.callbacks import Phase, PhaseCallback, PhaseContext
from torchvision.ops import box_convert
import plotly.graph_objs as go


class DetectionVisualizationCallback(PhaseCallback):
    """
    A callback that adds a visualization of a batch of segmentation predictions to context.sg_logger
    Attributes:
        freq: frequency (in epochs) to perform this callback.
        batch_idx: batch index to perform visualization for.
        last_img_idx_in_batch: Last image index to add to log. (default=-1, will take entire batch).
    """

    test_sequence_name = "test_det"

    def __init__(
        self,
        logger,
        phase: Phase,
        freq: int,
        id2label,
        batch_idxs=None,
        last_img_idx_in_batch: int = None,
        undo_preprocessing=None,
        threshold=None,
        metrics=False,
    ):
        super(DetectionVisualizationCallback, self).__init__(phase)
        if batch_idxs is None:
            batch_idxs = [0]
        self.freq = freq
        self.id2label = id2label
        self.batch_idxs = batch_idxs
        self.last_img_idx_in_batch = last_img_idx_in_batch
        self.undo_preprocesing = undo_preprocessing
        self.metrics = metrics
        self.threshold = threshold
        self.prefix = (
            "train"
            if phase == Phase.TRAIN_BATCH_END
            else "val"
            if phase == Phase.VALIDATION_BATCH_END
            else "test"
        )

    def __call__(self, context: PhaseContext):
        epoch = context.epoch if context.epoch is not None else 0
        if epoch % self.freq == 0 and context.batch_idx in self.batch_idxs:
            if hasattr(context.preds, "student_output"):  # is knowledge distillation
                preds = context.preds.student_output.clone()
            elif hasattr(context.preds, "main"):  # is composed output
                preds = {k: v for k, v in context.preds.main.items()}
            else:
                preds = {k: v for k, v in context.preds.items()}
            DetectionVisualization.visualize_batch(
                logger=context.sg_logger,
                image_tensor=context.inputs,
                preds=preds,
                targets=context.target,
                id2label=self.id2label,
                batch_name=context.batch_idx,
                undo_preprocessing_func=self.undo_preprocesing,
                prefix=self.prefix,
                names=context.input_name,
                iteration=context.epoch,
                threshold=self.threshold,
            )


class DetectionVisualization:
    """
    A helper class to visualize detections predicted by a network.
    """

    @staticmethod
    def visualize_batch(
        logger,
        image_tensor: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        id2label: Mapping[int, str],
        batch_name: Union[int, str],
        undo_preprocessing_func: Callable[[torch.Tensor], np.ndarray] = lambda x: x,
        use_plotly: bool = False,
        prefix: str = "",
        names: List[str] = None,
        iteration: int = 0,
        threshold: float = None,
    ):
        """
        A helper function to visualize detections predicted by a network:
        saves images into a given path with a name that is {batch_name}_{imade_idx_in_the_batch}.jpg, one batch per call.
        Colors are generated on the fly: uniformly sampled from color wheel to support all given classes.

        :param iteration:
        :param names:
        :param prefix:
        :param image_tensor:            rgb images, (B, H, W, 3)
        :param batch_name:              id of the current batch to use for image naming
        :param undo_preprocessing_func: a function to convert preprocessed images tensor into a batch of cv2-like images
        :param image_scale:             scale factor for output image
        """
        if isinstance(image_tensor, list):
            # Dealing with (text, image, attention_mask) tuple
            image_tensor = image_tensor[1]
        image_tensor = image_tensor.detach()
        if use_plotly:
            image_np = image_tensor.cpu().numpy()
        else:
            image_np = (
                undo_preprocessing_func(image_tensor).type(dtype=torch.uint8).cpu()
            )

        if names is None:
            names = [
                "_".join([prefix, "det", str(batch_name), str(i)])
                if prefix == "val"
                else "_".join([prefix, "det", str(batch_name * image_np.shape[0] + i)])
                for i in range(image_np.shape[0])
            ]
        else:
            names = [f"{prefix}_det_{name}" for name in names]

        for i in range(image_np.shape[0]):
            pred = {"boxes": preds["pred_boxes"][i], "logits": preds["logits"][i]}
            target = {
                "boxes": targets[i]["boxes"],
                "labels": targets[i]["class_labels"],
            }

            if threshold is not None:
                mask = pred["logits"].sigmoid().max(-1)[0] > threshold
                pred["boxes"] = pred["boxes"][mask]
                pred["logits"] = pred["logits"][mask]

            pred["labels"] = pred["logits"].argmax(-1)

            fig = DetectionVisualization._visualize_image(
                image_np[i], pred, target, id2label
            )
            if prefix == "val":
                logger.add_plotly_figure(names[i], fig, global_step=iteration)
            else:
                logger.add_plotly_figure(names[i], fig)

    @staticmethod
    def add_boxes(img, labels, id2label, in_fmt, color):
        boxes = labels["boxes"]
        classes = labels["labels"]
        boxes = boxes * torch.tensor(img.shape[-2:], device=boxes.device).repeat(2)
        boxes = box_convert(boxes, in_fmt=in_fmt, out_fmt="xyxy")

        rectangles = []
        annotations = []
        for bbox, label in zip(boxes, classes):
            x_min, y_min, x_max, y_max = bbox.cpu().detach().numpy()
            # Add the rectangle to the figure
            rectangles.append(
                go.layout.Shape(
                    type="rect",
                    x0=x_min,
                    y0=y_min,
                    x1=x_max,
                    y1=y_max,
                    line=dict(color=color, width=2),
                    fillcolor="rgba(0,0,0,0)",  # Transparent fill
                )
            )
            annotations.append(
                go.layout.Annotation(
                    text=id2label[label.item()],
                    x=x_min,
                    y=y_min,
                    showarrow=False,
                    font=dict(color="white", size=12),
                    bgcolor=color,
                    xanchor="left",
                    yanchor="bottom",
                )
            )
        return rectangles, annotations

    @staticmethod
    def _visualize_image(img, pred_boxes, target_boxes, id2label, in_fmt="cxcywh"):
        to_draw = img.permute(1, 2, 0).numpy()

        # Create the image object
        image = go.Image(z=to_draw)
        pred_rectangles, pred_annotations = DetectionVisualization.add_boxes(
            img, pred_boxes, id2label, in_fmt, color="red"
        )
        target_rectangles, target_annotations = DetectionVisualization.add_boxes(
            img, target_boxes, id2label, in_fmt, color="green"
        )

        fig = go.Figure(
            data=image,
            layout=go.Layout(
                shapes=pred_rectangles + target_rectangles,
                annotations=pred_annotations + target_annotations,
            ),
        )
        # Set the layout
        w, h = to_draw.shape[:2]
        fig.update_layout(
            title="Image with Bounding Boxes",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=w,
            height=h,
            xaxis_range=[0, w],
            yaxis_range=[0, h],
        )
        return fig
