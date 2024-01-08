from label_anything.visualization.visualize import get_image
from label_anything.logger.utils import (
    extract_polygons_from_tensor,
    crop_padding,
    resize_gt,
    take_image,
    extract_masks_dynamic,
)
from label_anything.utils.utils import log_every_n
import os
import math
import torch
import tempfile
import torch.nn.functional as F

from PIL import Image


def validate_polygon(polygon):
    return len(polygon) >= 6  # 3 points at least


class Logger:
    def __init__(
        self,
        experiment,
        tmp_dir: str,
        log_frequency: int = 100,
        train_image_log_frequency: int = 1000,
        val_image_log_frequency: int = 1000,
        test_image_log_frequency: int = 1000,
    ):
        self.experiment = experiment
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.log_frequency = log_frequency
        self.prefix_frequency_dict = {
            "train": train_image_log_frequency,
            "val": val_image_log_frequency,
            "test": test_image_log_frequency,
        }

    def __get_class_ids(self, classes):
        res_classes = []
        for c in classes:
            max_len = 0
            max_idx = 0
            for i, x in enumerate(c):
                max_len = max(max_len, len(x))
                if len(x) == max_len:
                    max_idx = i
            res_classes.append(list(c[max_idx]))
        return res_classes

    def log_batch(
        self,
        batch_idx,
        image_idx,
        batch_size,
        step,
        substitution_step,
        input_dict,
        gt,
        pred,
        dataset,
        dataset_names,
        phase,
    ):
        if log_every_n(image_idx, batch_size, self.prefix_frequency_dict[phase]):
            dataset.log_images = True
            return
        if dataset.log_images:
            categories = dataset.categories
            self.log_prompts(
                batch_idx=batch_idx,
                step=step,
                image_idx=image_idx,
                substitution_step=substitution_step,
                input_dict=input_dict,
                categories=categories,
                dataset_names=dataset_names,
                prefix=phase,
            )
            self.log_gt_pred(
                batch_idx=batch_idx,
                image_idx=image_idx,
                step=step,
                substitution_step=substitution_step,
                input_dict=input_dict,
                gt=gt,
                pred=pred,
                categories=categories,
                dataset_names=dataset_names,
                prefix=phase,
            )
            dataset.log_images = False

    def log_gt_pred(
        self,
        batch_idx,
        image_idx,
        step,
        substitution_step,
        input_dict,
        gt,
        pred,
        categories,
        dataset_names,
        prefix
    ):
        images = input_dict["images"]
        dims = input_dict["dims"]
        classes = self.__get_class_ids(input_dict["classes"])

        for b in range(gt.shape[0]):
            data_gt = []
            data_pred = []
            img = take_image(images[b, 0], dims[b, 0])
            image = get_image(img)
            cur_dataset_categories = categories[dataset_names[b]]

            sample_gt = gt[b, : dims[b, 0, 0], : dims[b, 0, 1]]
            sample_gt = F.one_hot(sample_gt, num_classes=len(classes[b]) + 1).permute(
                2, 0, 1
            )

            sample_pred = pred[b, :, : dims[b, 0, 0], : dims[b, 0, 1]]
            sample_pred = torch.argmax(sample_pred, dim=0)
            sample_pred = F.one_hot(
                sample_pred, num_classes=len(classes[b]) + 1
            ).permute(2, 0, 1)

            for c in range(1, sample_gt.shape[0]):
                label = cur_dataset_categories[classes[b][c - 1]]["name"]
                polygons_gt = extract_polygons_from_tensor(
                    sample_gt[c], should_resize=False
                )
                polygons_pred = extract_polygons_from_tensor(
                    sample_pred[c], should_resize=False
                )
                polygons_pred = [
                    polygon for polygon in polygons_pred if validate_polygon(polygon)
                ]
                data_gt.append(
                    {"points": polygons_gt, "label": f"gt-{label}", "score": None}
                )
                if polygons_pred:
                    data_pred.append(
                        {
                            "points": polygons_pred,
                            "label": f"pred-{label}",
                            "score": None,
                        }
                    )

            annotations = [{"name": "Ground truth", "data": data_gt}]
            if data_pred:
                annotations.append({"name": "Prediction", "data": data_pred})
            self.log_image(
                name=f"{prefix}_image_{image_idx}_sample_{b}_substep_{substitution_step}_gt_pred",
                image_data=image,
                annotations=annotations,
                metadata={
                    "batch_idx": batch_idx,
                    "image_idx": image_idx,
                    "sample_idx": b,
                    "substitution_step": substitution_step,
                    "type": "gt_pred",
                    "pred_bg_percent": torch.sum(sample_pred[0]).item()
                    / (sample_pred.shape[1] * sample_pred.shape[2]),
                    "phase": prefix,
                },
                step=step,
            )

    def log_prompts(
        self,
        batch_idx,
        image_idx,
        step,
        substitution_step,
        input_dict,
        categories,
        dataset_names,
        prefix
    ):
        images = input_dict["images"]
        all_masks = input_dict["prompt_masks"]
        all_boxes = input_dict["prompt_bboxes"]
        all_points = input_dict["prompt_points"]
        flags_masks = input_dict["flag_masks"]
        flags_boxes = input_dict["flag_bboxes"]
        flags_points = input_dict["flag_points"]
        classes = self.__get_class_ids(input_dict["classes"])

        for i in range(images.shape[0]):
            cur_dataset_categories = categories[dataset_names[i]]
            sample_images = images[i]
            for j in range(all_masks.shape[1]):
                image = get_image(sample_images[j + 1])
                data = []
                annotations = [{"name": "Ground truth", "data": data}]

                # log masks, boxes and points
                for c in range(1, input_dict["prompt_masks"].shape[2]):
                    if c > len(classes[i]):
                        break
                    mask = all_masks[i, j, c]
                    boxes = all_boxes[i, j, c]
                    points = all_points[i, j, c]
                    flag_mask = flags_masks[i, j, c]
                    flag_boxes = flags_boxes[i, j, c]
                    flag_points = flags_points[i, j, c]
                    label = cur_dataset_categories[classes[i][c - 1]]["name"]

                    if flag_mask == 1:
                        polygons = extract_polygons_from_tensor(mask)
                        # print(polygons)
                        data.append({"points": polygons, "label": label, "score": None})

                    boxes_log = []
                    for k in range(boxes.shape[0]):
                        if flag_boxes[k] == 1:
                            b = boxes[k].tolist()
                            # from x1, y1, x2, y2 to x1, y1, w, h
                            b[2] = b[2] - b[0]
                            b[3] = b[3] - b[1]
                            boxes_log.append(b)
                    if len(boxes_log) > 0:
                        data.append({"boxes": boxes_log, "label": label, "score": None})

                    positive_points_log = []
                    negative_points_log = []
                    for k in range(points.shape[0]):
                        if flag_points[k] != 0:
                            x, y = points[k].tolist()
                            ps = []
                            radius = 10

                            # Number of points
                            num_points = 20

                            # Calculate and print the coordinates of the 10 points
                            for z in range(num_points):
                                theta = 2 * math.pi * z / num_points
                                x_new = x + radius * math.cos(theta)
                                y_new = y + radius * math.sin(theta)
                                ps += [int(x_new), int(y_new)]

                            if flag_points[k] == 1:
                                positive_points_log.append(ps)
                            else:
                                negative_points_log.append(ps)
                    if positive_points_log:
                        data.append(
                            {
                                "points": positive_points_log,
                                "label": label,
                                "score": None,
                            }
                        )
                    if negative_points_log:
                        data.append(
                            {
                                "points": negative_points_log,
                                "label": f"Neg-{label}",
                                "score": None,
                            }
                        )
                self.log_image(
                    name=f"{prefix}_image_{image_idx}_sample_{i}_substep_{substitution_step}_prompts",
                    image_data=image,
                    annotations=annotations,
                    metadata={
                        "batch_idx": batch_idx,
                        "image_idx": image_idx,
                        "sample_idx": i,
                        "substitution_step": substitution_step,
                        "type": "prompt",
                        "phase": prefix,
                    },
                    step=step,
                )

    def log_image(
        self, name: str, image_data: Image, annotations=None, metadata=None, step=None
    ):
        tmp_path = tempfile.mktemp(suffix=".png", dir=self.tmp_dir)
        image_data.save(tmp_path)
        self.experiment.log_image(
            name=name,
            image_data=tmp_path,
            metadata=metadata,
            annotations=annotations,
            step=step,
        )
        os.remove(tmp_path)

    def log_metric(self, name, metric, epoch=None):
        self.experiment.log_metric(name, metric, epoch)

    def log_metrics(self, metrics, epoch=None):
        for name, metric in metrics.items():
            self.log_metric(name, metric, epoch)

    def log_parameter(self, name, parameter):
        self.experiment.log_parameter(
            name,
            parameter,
        )


"""
Example usage:
==============================================================
image:
- a path (string) to an image
- file-like containg image
- numpy matrix
- tensorflow tensor
- pytorch tensor
- list of tuple of values
- PIL image
$ logger.log_image("image_name", image)
==============================================================
"""
