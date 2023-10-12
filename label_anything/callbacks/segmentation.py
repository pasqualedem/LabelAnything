import os
import torch

import numpy as np

from typing import Union, Callable, List

from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from PIL import ImageColor, Image
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class SegmentationVisualizationCallback(PhaseCallback):
    test_sequence_name = "test_seg"
    """
    A callback that adds a visualization of a batch of segmentation predictions to context.sg_logger
    Attributes:
        freq: frequency (in epochs) to perform this callback.
        batch_idx: batch index to perform visualization for.
        last_img_idx_in_batch: Last image index to add to log. (default=-1, will take entire batch).
    """

    def __init__(
        self,
        logger,
        phase: Phase,
        freq: int,
        id2label,
        batch_idxs=None,
        last_img_idx_in_batch: int = None,
        undo_preprocessing=None,
        use_plotly=False,
        metrics=False,
        cmap=None,
    ):
        super(SegmentationVisualizationCallback, self).__init__(phase)
        if batch_idxs is None:
            batch_idxs = [0]
        self.freq = freq
        self.id2label = id2label
        self.batch_idxs = batch_idxs
        self.last_img_idx_in_batch = last_img_idx_in_batch
        self.undo_preprocesing = undo_preprocessing
        self.cmap = cmap
        self.use_plotly = use_plotly
        self.metrics = metrics
        self.prefix = (
            "train"
            if phase == Phase.TRAIN_EPOCH_END
            else "val"
            if phase == Phase.VALIDATION_BATCH_END
            else "test"
        )
        if phase == Phase.TEST_BATCH_END:
            logger.create_image_mask_sequence(f"{self.prefix}_seg")

    def __call__(self, context: PhaseContext):
        epoch = context.epoch if context.epoch is not None else 0
        if epoch % self.freq == 0 and context.batch_idx in self.batch_idxs:
            if hasattr(context.preds, "student_output"):  # is knowledge distillation
                preds = context.preds.student_output.clone()
            elif hasattr(context.preds, "main"):  # is composed output
                preds = context.preds.main.clone()
            else:
                preds = context.preds.clone()
            if self.metrics and self.use_plotly:
                metrics = None
            SegmentationVisualization.visualize_batch(
                logger=context.sg_logger,
                image_tensor=context.inputs,
                pred_mask=preds,
                target_mask=context.target,
                num_classes=self.id2label,
                batch_name=context.batch_idx,
                undo_preprocessing_func=self.undo_preprocesing,
                prefix=self.prefix,
                names=context.input_name,
                iteration=context.epoch,
                use_plotly=self.use_plotly,
                cmap=self.cmap,
            )
            if (
                self.prefix == "test"
                and context.batch_idx == self.batch_idxs[-1]
                and not self.use_plotly
            ):
                context.sg_logger.add_image_mask_sequence(f"{self.prefix}_seg")


class SegmentationVisualization:
    @staticmethod
    def _visualize_image(
        image_np: np.ndarray,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        classes,
        cmap=None,
    ):
        """

        :param image_np: numpy image
        :param pred_mask: (C, H, W) tensor of classes in one hot encoding
        :param target_mask: (H, W) tensor of classes
        :param classes:
        :return:
        """
        pred_mask = torch.tensor(pred_mask.copy())
        target_mask = torch.tensor(target_mask.copy())

        pred_mask = pred_mask.argmax(dim=0)

        if image_np.shape[0] < 3:
            image_np = torch.vstack(
                [
                    image_np,
                    torch.zeros(
                        (3 - image_np.shape[0], *image_np.shape[1:]), dtype=torch.uint8
                    ),
                ]
            )
        image_np = image_np[:3, :, :]  # Take only 3 bands if there are more
        image_np = np.moveaxis(image_np.numpy(), 0, -1)

        return image_np, {
            "predictions": {
                "mask_data": pred_mask.numpy(),
                "id2label": classes,
                "cmap": cmap,
            },
            "ground_truth": {
                "mask_data": target_mask.numpy(),
                "id2label": classes,
                "cmap": cmap,
            },
        }

    @staticmethod
    def visualize_batch(
        logger,
        image_tensor: torch.Tensor,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        num_classes,
        batch_name: Union[int, str],
        undo_preprocessing_func: Callable[[torch.Tensor], np.ndarray] = lambda x: x,
        use_plotly: bool = False,
        prefix: str = "",
        names: List[str] = None,
        iteration: int = 0,
        cmap=None,
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
        image_tensor = image_tensor.detach()
        if use_plotly:
            image_np = image_tensor.cpu().numpy()
        else:
            image_np = (
                undo_preprocessing_func(image_tensor).type(dtype=torch.uint8).cpu()
            )

        if names is None:
            names = [
                "_".join([prefix, "seg", str(batch_name), str(i)])
                if prefix == "val"
                else "_".join([prefix, "seg", str(batch_name * image_np.shape[0] + i)])
                for i in range(image_np.shape[0])
            ]
        else:
            names = [f"{prefix}_seg_{name}" for name in names]

        for i in range(image_np.shape[0]):
            preds = pred_mask[i].detach().cpu().numpy()
            targets = target_mask[i].detach().cpu().numpy()

            if use_plotly:
                fig = SegmentationVisualization.visualize_with_plotly(
                    image_np[i], preds, targets, num_classes
                )
                if prefix == "val":
                    logger.add_plotly_figure(names[i], fig, global_step=iteration)
                else:
                    logger.add_plotly_figure(names[i], fig)
            else:
                img, mask_dict = SegmentationVisualization._visualize_image(
                    image_np[i], preds, targets, num_classes, cmap
                )
                if prefix == "val":
                    logger.add_mask(names[i], img, mask_dict, global_step=iteration)
                else:
                    logger.add_image_mask_to_sequence(
                        SegmentationVisualizationCallback.test_sequence_name,
                        names[i],
                        img,
                        mask_dict,
                    )

    @staticmethod
    def visualize_with_plotly(
        image: np.ndarray, pred_mask: torch.Tensor, target_mask: torch.Tensor, classes
    ):
        """

        :param image: numpy image
        :param pred_mask: (C, H, W) tensor of classes in one hot encoding
        :param target_mask: (H, W) tensor of classes
        :param classes:
        :return:
        """
        n_plots = image.shape[0] + 2
        cols = min(n_plots, 4)
        rows = int(np.ceil(n_plots / cols))
        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True)
        pred_mask = torch.tensor(pred_mask.copy())
        target_mask = torch.tensor(target_mask.copy())

        for i in range(image.shape[0]):
            row = i // cols + 1
            col = i % cols + 1
            trace = go.Heatmap(
                z=image[i],
                showlegend=False,
                colorscale="viridis",
                showscale=False,
                name=f"channel_{i}",
            )
            fig.add_trace(trace, row=row, col=col)

        fig.add_trace(
            go.Heatmap(
                z=pred_mask.argmax(dim=0).numpy(),
                showlegend=False,
                showscale=False,
                name="preds",
                zmin=0,
                zmax=len(classes) - 1,
            ),
            row=(n_plots - 2) // cols + 1,
            col=(n_plots - 2) % cols + 1,
        )
        fig.add_trace(
            go.Heatmap(
                z=target_mask.numpy(),
                showlegend=False,
                showscale=False,
                name="gt",
                zmin=0,
                zmax=len(classes) - 1,
            ),
            row=(n_plots - 1) // cols + 1,
            col=(n_plots - 1) % cols + 1,
        )

        min_size = 192
        width = target_mask.shape[1]
        height = target_mask.shape[0]
        if width < min_size:
            ratio = width / height
            width = min_size
            height = min_size // ratio

        fig.update_layout(
            width=width * cols,
            height=height * rows,
            margin=dict(t=100, b=100, l=50, r=50),
        )

        return fig


class SaveSegmentationPredictionsCallback(PhaseCallback):
    def __init__(self, phase, path, num_classes):
        super(SaveSegmentationPredictionsCallback, self).__init__(phase)
        self.path = path
        self.num_classes = num_classes

        os.makedirs(self.path, exist_ok=True)
        colors = ["blue", "green", "red"]
        self.colors = []
        for color in colors:
            if isinstance(color, str):
                color = ImageColor.getrgb(color)
            self.colors.append(torch.tensor(color, dtype=torch.uint8))

    def __call__(self, context: PhaseContext):
        for prediction, input_name in zip(context.preds, context.input_name):
            path = os.path.join(self.path, input_name)
            prediction = prediction.detach().cpu()
            masks = torch.concat(
                [
                    (prediction.argmax(0) == cls).unsqueeze(0)
                    for cls in range(self.num_classes)
                ]
            )

            img_to_draw = torch.zeros(*prediction.shape[-2:], 3, dtype=torch.uint8)
            # TODO: There might be a way to vectorize this
            for mask, color in zip(masks, self.colors):
                img_to_draw[mask] = color

            Image.fromarray(img_to_draw.numpy()).save(path)
