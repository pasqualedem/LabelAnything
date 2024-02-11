from copy import deepcopy
from label_anything.visualization.visualize import get_image
from label_anything.logger.utils import (
    extract_polygons_from_tensor,
    get_tmp_dir,
    take_image,
)
from label_anything.logger.text_logger import get_logger
from label_anything.logger.abstract_logger import AbstractLogger
from label_anything.utils.utils import log_every_n
import os
import math
import time
import torch
import tempfile
import comet_ml
import torch.nn.functional as F

from datetime import datetime as dt
from PIL import Image
from comet_ml import OfflineExperiment, Experiment
from comet_ml.offline_utils import create_experiment_archive
from comet_ml.offline import OFFLINE_EXPERIMENT_END
from accelerate import Accelerator


logger = get_logger(__name__)


def validate_polygon(polygon):
    return len(polygon) >= 6  # 3 points at least


def comet_experiment(accelerator: Accelerator, params: dict):
    comet_params = params.get("logger", {}).pop("comet", {})
    comet_information = {
        "api_key": os.getenv("COMET_API_KEY"),
        "project_name": params["experiment"]["name"],
        **comet_params,
    }
    global logger
    logger_params = deepcopy(params.get("logger", {}))
    tmp_dir = get_tmp_dir()
    if tmp_dir:
        logger.info(
            f"Using {tmp_dir} as temporary directory from environment variables"
        )
    else:
        tmp_dir = logger_params.get("tmp_dir", None)
        logger.info(
            f"No temporary directory found in environment variables, using {tmp_dir} for images"
        )
    os.makedirs(tmp_dir, exist_ok=True)
    tags = logger_params.pop("tags", [])
    comet_logger = CometLogger(accelerator=accelerator, comet_information=comet_information, **logger_params)

    comet_logger.add_tags(tags)
    comet_logger.log_parameters(params)
    
    return comet_logger


class CometLogger(AbstractLogger):
    def __init__(
        self,
        accelerator: Accelerator,
        tmp_dir: str,
        log_frequency: int = 100,
        train_image_log_frequency: int = 1000,
        val_image_log_frequency: int = 1000,
        test_image_log_frequency: int = 1000,
        experiment_save_delta: int = None,
        
        comet_information: dict = {},
    ):
        if comet_information.pop("offline"):
            offdir = comet_information.pop("offline_directory", None)
            experiment = comet_ml.OfflineExperiment(
                offline_directory=offdir, **comet_information
            )
        else:
            experiment = comet_ml.Experiment(**comet_information)
        comet_ml.init(comet_information)

        super().__init__(
            experiment,
            accelerator,
            tmp_dir,
            log_frequency,
            train_image_log_frequency,
            val_image_log_frequency,
            test_image_log_frequency,
            experiment_save_delta,
        )
        self.train = self.experiment.train
        self.validate = self.experiment.validate
        self.test = self.experiment.test
        
    def add_tags(self, tags):
        self.experiment.add_tags(tags)
        
    def log_parameters(self, params):
        self.experiment.log_parameters(params)

    def log_gt_pred(
        self,
        batch_idx,
        image_idx,
        epoch,
        step,
        substitution_step,
        input_dict,
        images,
        gt,
        pred,
        categories,
        dataset_names,
        prefix,
    ):
        dims = input_dict["dims"]
        classes = self._get_class_ids(input_dict["classes"])

        for b in range(gt.shape[0]):
            data_gt = []
            data_pred = []
            image = get_image(take_image(images[b], dims[b, 0]))
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
                    "epoch": epoch,
                    "step": step,
                    "pred_bg_percent": torch.sum(sample_pred[0]).item()
                    / (sample_pred.shape[1] * sample_pred.shape[2]),
                    "phase": prefix,
                },
                step=epoch,
            )

    def log_prompts(
        self,
        batch_idx,
        image_idx,
        epoch,
        step,
        substitution_step,
        input_dict,
        images,
        categories,
        dataset_names,
        prefix,
    ):
        all_masks = input_dict["prompt_masks"]
        all_boxes = input_dict["prompt_bboxes"]
        all_points = input_dict["prompt_points"]
        flags_masks = input_dict["flag_masks"]
        flags_boxes = input_dict["flag_bboxes"]
        flags_points = input_dict["flag_points"]
        classes = self._get_class_ids(input_dict["classes"])

        for i in range(len(images)):
            cur_dataset_categories = categories[dataset_names[i]]
            sample_images = images[i]
            for j in range(all_masks.shape[1]):
                image = get_image(sample_images[j])
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
                        "epoch": epoch,
                        "step": step,
                        "phase": prefix,
                    },
                    step=epoch,
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
        # os.remove(tmp_path)

    def log_metric(self, name, metric, epoch=None):
        if self.accelerator.is_local_main_process:
            self.experiment.log_metric(name, metric, epoch)

    def log_metrics(self, metrics, epoch=None):
        for name, metric in metrics.items():
            self.log_metric(name, metric, epoch)

    def log_parameter(self, name, parameter):
        self.experiment.log_parameter(
            name,
            parameter,
        )

    def log_asset_folder(self, folder, step=None):
        self.experiment.log_asset_folder(folder, step)

    def save_experiment_timed(self):
        """
        Save the experiment every `self.time_delta` seconds
        """
        if self.experiment_save_delta is None:
            return
        if type(self.experiment) != OfflineExperiment:
            return
        if time.time() - self.start_time > self.experiment_save_delta:
            logger.info(
                f"Saving partial experiment as it has been running for more than {self.experiment_save_delta} seconds"
            )
            self.save_experiment()
            self.start_time = time.time()

    def save_experiment(self):
        if type(self.experiment) != OfflineExperiment:
            return
        logger.info("Waiting for all processes to finish for saving partial experiment")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process:
            self.experiment._write_experiment_meta_file()
            self.experiment.add_tag("Partial")

            zip_file_filename, _ = create_experiment_archive(
                offline_directory=self.experiment.offline_directory,
                offline_archive_file_name=self.experiment._get_offline_archive_file_name(),
                data_dir=self.experiment.tmpdir,
                logger=logger,
            )
            # Display the full command to upload the offline experiment
            logger.info(
                f"Partial experiment saved at time {dt.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.info(OFFLINE_EXPERIMENT_END, zip_file_filename)
        logger.info("Waiting for main process to finish saving partial experiment")
        self.accelerator.wait_for_everyone()
        logger.info("Finished saving partial experiment")

    def end(self):
        if "Partial" in self.experiment.tags:
            logger.info("Removing partial tag from experiment")
            self.experiment.tags.remove("Partial")
        self.experiment.end()


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
