from contextlib import contextmanager
from copy import deepcopy
import math
import os
from typing import Optional, Union, Any

import pandas as pd
import numpy as np

import torch.nn.functional as F

import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt

from label_anything.data.utils import to_global_multiclass
from label_anything.logger.text_logger import get_logger
from accelerate import Accelerator
from label_anything.logger.utils import get_tmp_dir, take_image
from label_anything.utils.utils import log_every_n, write_yaml

from label_anything.visualization.visualize import get_image


logger = get_logger(__name__)

WANDB_ID_PREFIX = "wandb_id."
WANDB_INCLUDE_FILE_NAME = ".wandbinclude"


def main_process_only(func):
    def wrapper(instance, *args, **kwargs):
        accelerator = instance.accelerator
        if accelerator.is_local_main_process:
            return func(instance, *args, **kwargs)

    return wrapper


def wandb_tracker(accelerator: Accelerator, params: dict):
    logger_params = deepcopy(params.get("logger", {}))
    wandb_information = {
        "accelerator": accelerator,
        "project_name": params["experiment"]["name"],
        "group": params["experiment"].get("group", None),
    }
    global logger
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
    logger_params["tmp_dir"] = tmp_dir

    wandb_logger = WandBLogger(**wandb_information, **logger_params)
    wandb_logger.log_parameters(params)
    wandb_logger.add_tags(logger_params.get("tags", ()))

    return wandb_logger


class WandBLogger:
    MAX_CLASSES = 100000  # For negative classes

    def __init__(
        self,
        project_name: str,
        accelerator: Accelerator,
        tmp_dir: str,
        resume: bool = False,
        offline_directory: str = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = True,
        save_logs_remote: bool = True,
        entity: Optional[str] = None,
        save_code: bool = False,
        tags=None,
        run_id=None,
        resume_checkpoint_type: str = "best",
        group=None,
        log_frequency: int = 100,
        train_image_log_frequency: int = 1000,
        val_image_log_frequency: int = 1000,
        test_image_log_frequency: int = 1000,
        ignore_index = -100
    ):
        """
        :param project_name: The WandB project name.
        :param accelerator: Accelerator instance.
        :param tmp_dir: Temporary directory for local data.
        :param resume: Whether to resume a previous run.
        :param offline_directory: Local directory for offline WandB logging.
        :param save_checkpoints_remote: Save checkpoints in remote storage.
        :param save_tensorboard_remote: Save TensorBoard logs in remote storage.
        :param save_logs_remote: Save general logs in remote storage.
        :param entity: WandB entity for project ownership.
        :param save_code: Save the current code to WandB.
        :param tags: List of tags for the WandB run.
        :param run_id: Unique ID to resume a WandB run.
        :param resume_checkpoint_type: Type of checkpoint for resuming.
        :param group: WandB group name for runs.
        :param log_frequency: Frequency of general logging.
        :param train_image_log_frequency: Frequency for logging train images.
        :param val_image_log_frequency: Frequency for logging validation images.
        :param test_image_log_frequency: Frequency for logging test images.
        """
        
        # Handle WandB-specific resume logic
        tracker_resume = "must" if resume else None
        self.resume = tracker_resume
        resume = run_id is not None
        if not tracker_resume and resume:
            tags = (tags or []) + ["resume", run_id]

        # Offline directory setup for WandB
        self.accelerator_state_dir = None
        if offline_directory:
            os.makedirs(offline_directory, exist_ok=True)
            os.environ["WANDB_ARTIFACT_LOCATION"] = offline_directory
            os.environ["WANDB_ARTIFACT_DIR"] = offline_directory
            os.environ["WANDB_CACHE_DIR"] = offline_directory
            os.environ["WANDB_CONFIG_DIR"] = offline_directory
            os.environ["WANDB_DATA_DIR"] = offline_directory

        # Resume WandB run if applicable
        if resume:
            self._resume(offline_directory, run_id, checkpoint_type=resume_checkpoint_type)

        # Initialize WandB experiment only on local main process
        experiment = None
        if accelerator.is_local_main_process:
            experiment = wandb.init(
                project=project_name,
                entity=entity,
                resume=tracker_resume,
                id=run_id if tracker_resume else None,
                tags=tags,
                dir=offline_directory,
                group=group,
            )
            logger.info(f"wandb run id  : {experiment.id}")
            logger.info(f"wandb run name: {experiment.name}")
            logger.info(f"wandb run dir : {experiment.dir}")
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("validate/step")
            wandb.define_metric("validate/*", step_metric="validate/step")

        self.experiment = experiment
        self.accelerator = accelerator
        self.tmp_dir = tmp_dir
        self.local_dir = experiment.dir if hasattr(experiment, "dir") else None
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.log_frequency = log_frequency
        self.prefix_frequency_dict = {
            "train": train_image_log_frequency,
            "val": val_image_log_frequency,
            "test": test_image_log_frequency,
        }
        
        # Additional WandBLogger attributes
        if save_code:
            self._save_code()
        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        self.save_logs_wandb = save_logs_remote
        self.context = ""
        self.sequences = {}
        self.ignore_index = ignore_index
                
    def _resume(self, offline_directory, run_id, checkpoint_type="latest"):
        if not offline_directory:
            offline_directory = "."
        wandb_dir = os.path.join(offline_directory, "wandb")
        runs = os.listdir(wandb_dir)
        runs = sorted(list(filter(lambda x: run_id in x, runs)))
        if len(runs) == 0:
            raise ValueError(f"Run {run_id} not found in {wandb_dir}")
        if len(runs) > 1:
            logger.warning(f"Multiple runs found for {run_id} in {wandb_dir}")
            for run in runs:
                logger.warning(run)
            logger.warning(f"Using {runs[0]}")
        run = runs[0]
        self.accelerator_state_dir = os.path.join(wandb_dir, run, "files", checkpoint_type)
        logger.info(f"Resuming from {self.accelerator_state_dir}")
        
    def _save_code(self):
        """
        Save the current code to wandb.
        If a file named .wandbinclude is avilable in the root dir of the project the settings will be taken from the file.
        Otherwise, all python file in the current working dir (recursively) will be saved.
        File structure: a single relative path or a single type in each line.
        i.e:

        src
        tests
        examples
        *.py
        *.yaml

        The paths and types in the file are the paths and types to be included in code upload to wandb
        """
        base_path, paths, types = self._get_include_paths()

        if len(types) > 0:

            def func(path):
                for p in paths:
                    if path.startswith(p):
                        for t in types:
                            if path.endswith(t):
                                return True
                return False

            include_fn = func
        else:
            include_fn = lambda path: path.endswith(".py")

        if base_path != ".":
            wandb.run.log_code(base_path, include_fn=include_fn)
        else:
            wandb.run.log_code(".", include_fn=include_fn)

    @main_process_only
    def log_parameters(self, config: dict = None):
        wandb.config.update(config, allow_val_change=self.resume)
        tmp = os.path.join(self.local_dir, "config.yaml")
        write_yaml(config, tmp)
        self.add_file("config.yaml")
        
    @main_process_only
    def add_tags(self, tags):
        wandb.run.tags = wandb.run.tags + tuple(tags)

    @main_process_only
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        wandb.log(data={tag: scalar_value}, step=global_step)

    @main_process_only
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        for name, value in tag_scalar_dict.items():
            if isinstance(value, dict):
                tag_scalar_dict[name] = value["value"]
        wandb.log(data=tag_scalar_dict, step=global_step)

    @main_process_only
    def add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.array, Image.Image],
        data_format="CHW",
        global_step: int = 0,
    ):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        wandb.log(data={tag: wandb.Image(image, caption=tag)}, step=global_step)

    @main_process_only
    def add_images(
        self,
        tag: str,
        images: Union[torch.Tensor, np.array],
        data_format="NCHW",
        global_step: int = 0,
    ):
        wandb_images = []
        for im in images:
            if isinstance(im, torch.Tensor):
                im = im.cpu().detach().numpy()

            if im.shape[0] < 5:
                im = im.transpose([1, 2, 0])
            wandb_images.append(wandb.Image(im))
        wandb.log({tag: wandb_images}, step=global_step)

    @main_process_only
    def add_video(
        self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0
    ):
        if video.ndim > 4:
            for index, vid in enumerate(video):
                self.add_video(tag=f"{tag}_{index}", video=vid, global_step=global_step)
        else:
            if isinstance(video, torch.Tensor):
                video = video.cpu().detach().numpy()
            wandb.log({tag: wandb.Video(video, fps=4)}, step=global_step)

    @main_process_only
    def add_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.array],
        bins: str,
        global_step: int = 0,
    ):
        wandb.log({tag: wandb.Histogram(values, num_bins=bins)}, step=global_step)

    @main_process_only
    def add_plot(self, tag: str, values: pd.DataFrame, xtitle, ytitle, classes_marker):
        table = wandb.Table(columns=[classes_marker, xtitle, ytitle], dataframe=values)
        plt = wandb.plot_table(
            tag,
            table,
            {"x": xtitle, "y": ytitle, "class": classes_marker},
            {
                "title": tag,
                "x-axis-title": xtitle,
                "y-axis-title": ytitle,
            },
        )
        wandb.log({tag: plt})

    @main_process_only
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        wandb.log({tag: text_string}, step=global_step)

    @main_process_only
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        wandb.log({tag: figure}, step=global_step)

    @main_process_only
    def add_mask(self, tag: str, image, mask_dict, global_step: int = 0):
        wandb.log({tag: wandb.Image(image, masks=mask_dict)}, step=global_step)

    @main_process_only
    def add_table(self, tag, data, columns, rows):
        if isinstance(data, torch.Tensor):
            data = [[x.item() for x in row] for row in data]
        table = wandb.Table(data=data, rows=rows, columns=columns)
        wandb.log({tag: table})

    @main_process_only
    def end(self):
        wandb.finish()

    @main_process_only
    def add_file(self, file_name: str = None):
        wandb.save(
            glob_str=os.path.join(self.local_dir, file_name),
            base_path=self.local_dir,
            policy="now",
        )

    @main_process_only
    def add_summary(self, metrics: dict):
        wandb.summary.update(metrics)

    @main_process_only
    def upload(self):
        if self.save_tensorboard_wandb:
            wandb.save(
                glob_str=self._get_tensorboard_file_name(),
                base_path=self.local_dir,
                policy="now",
            )

        if self.save_logs_wandb:
            wandb.save(
                glob_str=self.log_file_path, base_path=self.local_dir, policy="now"
            )

    @main_process_only
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"

        path = os.path.join(self.local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints_wandb:
            if self.s3_location_available:
                self.model_checkpoints_data_interface.save_remote_checkpoints_file(
                    self.experiment_name, self.local_dir, name
                )
            wandb.save(glob_str=path, base_path=self.local_dir, policy="now")

    @main_process_only
    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError as e:
            logger.warning("tensorboard file could not be located for ")
            return None

        return tb_file_path

    @main_process_only
    def _get_wandb_id(self):
        for file in os.listdir(self.local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                return file.replace(WANDB_ID_PREFIX, "")

    @main_process_only
    def _set_wandb_id(self, id):
        for file in os.listdir(self.local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                os.remove(os.path.join(self.local_dir, file))

    @main_process_only
    def add(self, tag: str, obj: Any, global_step: int = None):
        pass

    @main_process_only
    def _get_include_paths(self):
        """
        Look for .wandbinclude file in parent dirs and return the list of paths defined in the file.

        file structure is a single relative (i.e. src/) or a single type (i.e *.py)in each line.
        the paths and types in the file are the paths and types to be included in code upload to wandb
        :return: if file exists, return the list of paths and a list of types defined in the file
        """

        wandb_include_file_path = self._search_upwards_for_file(WANDB_INCLUDE_FILE_NAME)
        if wandb_include_file_path is not None:
            with open(wandb_include_file_path) as file:
                lines = file.readlines()

            base_path = os.path.dirname(wandb_include_file_path)
            paths = []
            types = []
            for line in lines:
                line = line.strip().strip("/n")
                if line == "" or line.startswith("#"):
                    continue

                if line.startswith("*."):
                    types.append(line.replace("*", ""))
                else:
                    paths.append(os.path.join(base_path, line))
            return base_path, paths, types

        return ".", [], []

    @staticmethod
    def _search_upwards_for_file(file_name: str):
        """
        Search in the current directory and all directories above it for a file of a particular name.
        :param file_name: file name to look for.
        :return: pathlib.Path, the location of the first file found or None, if none was found
        """

        try:
            cur_dir = os.getcwd()
            while cur_dir != "/":
                if file_name in os.listdir(cur_dir):
                    return os.path.join(cur_dir, file_name)
                else:
                    cur_dir = os.path.dirname(cur_dir)
        except RuntimeError as e:
            return None

        return None
    
    def _get_class_ids(self, classes):
        res_classes = []
        for c in classes:
            res_classes.append(sorted(list(set(sum(c, [])))))
        return res_classes

    @main_process_only
    def log_batch(
        self,
        batch_idx,
        image_idx,
        batch_size,
        epoch,
        step,
        substitution_step,
        input_dict,
        input_shape,
        gt,
        pred,
        dataset,
        dataset_names,
        phase,
        run_idx,
    ):
        if log_every_n(batch_idx, self.prefix_frequency_dict[phase]) and (run_idx == 0 or run_idx is None):
            query_images = [
                dataset.load_and_preprocess_images(dataset_name, [ids[0]])[0]
                for dataset_name, ids in zip(dataset_names, input_dict["image_ids"])
            ]
            example_images = [
                dataset.load_and_preprocess_images(dataset_name, ids[1:])
                for dataset_name, ids in zip(dataset_names, input_dict["image_ids"])
            ]
            categories = dataset.categories
            # sequence_name = f"image_{image_idx}_substep_{substitution_step}"
            sequence_name = filter(lambda x: "predictions" in x, self.sequences.keys())
            sequence_name = next(sequence_name, None)
            self.log_prompts(
                batch_idx=batch_idx,
                epoch=epoch,
                step=step,
                image_idx=image_idx,
                substitution_step=substitution_step,
                input_dict=input_dict,
                images=example_images,
                categories=categories,
                dataset_names=dataset_names,
                prefix=phase,
                sequence_name=sequence_name,
            )
            self.log_gt_pred(
                batch_idx=batch_idx,
                image_idx=image_idx,
                epoch=epoch,
                step=step,
                substitution_step=substitution_step,
                input_dict=input_dict,
                input_shape=input_shape,
                images=query_images,
                gt=gt,
                pred=pred,
                categories=categories,
                dataset_names=dataset_names,
                prefix=phase,
                sequence_name=sequence_name,
            )
            
    @main_process_only
    def log_test_prediction(
        self,
        batch_idx,
        input_dict,
        gt,
        pred,
        input_shape,
        id2classes,
        dataset_name,
    ):
        if not log_every_n(batch_idx, self.prefix_frequency_dict["test"]):
            return
        dims = input_dict["dims"]
        images = input_dict["images"][:, 0]

        for b in range(gt.shape[0]):
            image = get_image(take_image(images[b], dims[b], input_shape=input_shape))

            sample_gt = gt[b, : dims[b, 0], : dims[b, 1]].detach().cpu().numpy()

            sample_pred = pred[b, :, : dims[b, 0], : dims[b, 1]]
            sample_pred = torch.argmax(sample_pred, dim=0).detach().cpu().numpy()

            wandb_image = wandb.Image(
                image,
                masks={
                    "ground_truth": {
                        "mask_data": sample_gt,
                        "class_labels": id2classes,
                    },
                    "prediction": {
                        "mask_data": sample_pred,
                        "class_labels": id2classes,
                    },
                },
                classes=[
                    {"id": c, "name": name} for c, name in id2classes.items()
                ],
            )

            self.add_image_to_sequence(
                dataset_name,
                f"image_{batch_idx}_sample_{b}",
                wandb_image,
            )

    @main_process_only
    def log_gt_pred(
        self,
        batch_idx,
        image_idx,
        epoch,
        step,
        substitution_step,
        input_dict,
        input_shape,
        images,
        gt,
        pred,
        categories,
        dataset_names,
        prefix,
        sequence_name,
    ):
        dims = input_dict["dims"]
        classes = self._get_class_ids(input_dict["classes"])
        pred, gt = to_global_multiclass(input_dict["classes"], categories, pred, gt, compact=False)

        for b in range(gt.shape[0]):
            image = get_image(take_image(images[b], dims[b, 0], input_shape=input_shape))
            cur_dataset_categories = categories[dataset_names[b]]
            cur_sample_categories = {
                c: cur_dataset_categories[c]["name"]
                for c in classes[b]
            }
            cur_sample_categories[0] = "background"

            sample_gt = gt[b, : dims[b, 0, 0], : dims[b, 0, 1]].detach().cpu().numpy()
            # Remove ignore index
            sample_gt[sample_gt == self.ignore_index] = 0

            sample_pred = pred[b, : dims[b, 0, 0], : dims[b, 0, 1]].detach().cpu().numpy()

            wandb_image = wandb.Image(
                image,
                masks={
                    "ground_truth": {
                        "mask_data": sample_gt,
                        "class_labels": cur_sample_categories,
                    },
                    "prediction": {
                        "mask_data": sample_pred,
                        "class_labels": cur_sample_categories,
                    },
                },
                classes=[
                    {"id": c, "name": name} for c, name in cur_sample_categories.items()
                ],
            )

            self.add_image_to_sequence(
                sequence_name,
                f"image_{image_idx}_sample_{b}_substep_{substitution_step}_gt_pred",
                wandb_image,
                metadata=[epoch, dataset_names[b]],
            )
            
    @main_process_only
    def log_test_prompts(
        self,
        input_dict,
        id2classes,
        dataset_name,
    ):
        sequence_name = f"{dataset_name}_prompts"
        self.create_image_sequence(sequence_name)
        all_masks = (
            input_dict["prompt_masks"].argmax(dim=1)
            if input_dict["prompt_masks"] is not None
            else None
        )
        all_boxes = input_dict["prompt_bboxes"]
        all_points = input_dict["prompt_points"]
        flags_masks = input_dict["flag_masks"]
        flags_boxes = input_dict["flag_bboxes"]
        flags_points = input_dict["flag_points"]
        images = input_dict["images"]

        for j in range(all_masks.shape[0]):
            image = get_image(images[j])
            # log masks, boxes and points
            points_data = []
            box_data = []
            for c in range(1, input_dict["prompt_masks"].shape[1]):
                if c > len(id2classes):
                    break
                boxes = all_boxes[j, c]
                points = all_points[j, c]
                flag_boxes = flags_boxes[j, c]
                flag_points = flags_points[j, c]
                label = id2classes[c]

                for k in range(boxes.shape[0]):
                    if flag_boxes[k] == 1:
                        box = boxes[k].tolist()
                        box = {
                            "position": {
                                "minX": box[0],
                                "minY": box[1],
                                "maxX": box[2],
                                "maxY": box[3],
                            },
                            "class_id": c,
                            "box_caption": f"{label}",
                            "domain": "pixel",
                        }
                        box_data.append(box)

                for k in range(points.shape[0]):
                    if flag_points[k] != 0:
                        x, y = points[k].tolist()
                        # point is one pixel bbox
                        if flag_points[k] == 1:
                            point_label = label
                            class_id = c
                        else:
                            point_label = f"Neg-{label}"
                            class_id = self.MAX_CLASSES + c

                        box = {
                            "position": {
                                "minX": x,
                                "minY": y,
                                "maxX": x + 10,
                                "maxY": y + 10,
                            },
                            "class_id": class_id,
                            "box_caption": "",
                            "domain": "pixel",
                        }
                        points_data.append(box)

            masks = None
            if flags_masks[j].sum() > 0:
                cur_mask = all_masks[j].unsqueeze(0).unsqueeze(0).float()
                masks = {
                    "ground_truth": {
                        "mask_data": F.interpolate(
                            cur_mask,
                            images[j].shape[-2:],
                        )
                        .squeeze()
                        .cpu()
                        .numpy(),
                        "class_labels": id2classes,
                    }
                }
            boxes = {}
            if len(box_data) > 0:
                boxes["boxes"] = {
                    "box_data": box_data,
                    "class_labels": id2classes,
                }
            if len(points_data) > 0:
                boxes["points"] = {
                    "box_data": points_data,
                    "class_labels": id2classes,
                }
            boxes = None if len(boxes) == 0 else boxes

            wandb_image = wandb.Image(
                image,
                masks=masks,
                boxes=boxes,
                classes=[
                    {"id": c, "name": name}
                    for c, name in id2classes.items()
                ],
            )

            self.add_image_to_sequence(
                sequence_name,
                f"image_{j}_prompts",
                wandb_image,
            )
        self.add_image_sequence(sequence_name)

    @main_process_only
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
        sequence_name,
    ):
        all_masks = (
            to_global_multiclass(input_dict["classes"], categories, input_dict["prompt_masks"].argmax(dim=2), compact=False)[0]
            if input_dict["prompt_masks"] is not None
            else None
        )
        all_boxes = input_dict["prompt_bboxes"]
        all_points = input_dict["prompt_points"]
        flags_masks = input_dict["flag_masks"]
        flags_boxes = input_dict["flag_bboxes"]
        flags_points = input_dict["flag_points"]
        classes = self._get_class_ids(input_dict["classes"])

        for i in range(len(images)):
            cur_dataset_categories = categories[dataset_names[i]]
            cur_sample_categories = {
                c: cur_dataset_categories[c]["name"]
                for c in classes[i]
            }
            cur_sample_categories[0] = "background"
            cur_sample_categories_dict = dict(enumerate(sorted(cur_sample_categories)))
            sample_images = images[i]
            for j in range(all_masks.shape[1]):

                mask_sample_categories = torch.argwhere(flags_masks[i, j])[:, 0].tolist()
                mask_sample_categories = {
                    cur_sample_categories_dict[k]: cur_sample_categories[cur_sample_categories_dict[k]] for k in mask_sample_categories
                }
                point_sample_categories = {}
                box_sample_categories = {}

                image = get_image(sample_images[j])
                # log masks, boxes and points
                points_data = []
                box_data = []
                for c in range(1, input_dict["prompt_masks"].shape[2]):
                    if c > len(classes[i]):
                        break
                    boxes = all_boxes[i, j, c]
                    points = all_points[i, j, c]
                    flag_boxes = flags_boxes[i, j, c]
                    flag_points = flags_points[i, j, c]
                    label = cur_sample_categories[cur_sample_categories_dict[c]]

                    for k in range(boxes.shape[0]):
                        if flag_boxes[k] == 1:
                            box = boxes[k].tolist()
                            box = {
                                "position": {
                                    "minX": box[0],
                                    "minY": box[1],
                                    "maxX": box[2],
                                    "maxY": box[3],
                                },
                                "class_id": cur_sample_categories_dict[c],
                                "box_caption": f"{label}",
                                "domain": "pixel",
                            }
                            box_sample_categories[cur_sample_categories_dict[c]] = label
                            box_data.append(box)

                    for k in range(points.shape[0]):
                        if flag_points[k] != 0:
                            x, y = points[k].tolist()
                            # point is one pixel bbox
                            if flag_points[k] == 1:
                                point_label = label
                                class_id = c
                            else:
                                point_label = f"Neg-{label}"
                                class_id = self.MAX_CLASSES + c
                            point_sample_categories[cur_sample_categories_dict[class_id]] = point_label
                            box = {
                                "position": {
                                    "minX": x,
                                    "minY": y,
                                    "maxX": x + 10,
                                    "maxY": y + 10,
                                },
                                "class_id": cur_sample_categories_dict[class_id],
                                "box_caption": "",
                                "domain": "pixel",
                            }
                            points_data.append(box)

                masks = None
                if flags_masks[i, j].sum() > 0:
                    cur_mask = all_masks[i, j].unsqueeze(0).unsqueeze(0).float()
                    masks = {
                        "ground_truth": {
                            "mask_data": F.interpolate(
                                cur_mask,
                                sample_images[j].shape[-2:],
                            )
                            .squeeze()
                            .cpu()
                            .numpy(),
                            "class_labels": mask_sample_categories,
                        }
                    }
                boxes = {}
                if box_data:
                    boxes["boxes"] = {
                        "box_data": box_data,
                        "class_labels": box_sample_categories,
                    }
                if points_data:
                    boxes["points"] = {
                        "box_data": points_data,
                        "class_labels": point_sample_categories,
                    }
                boxes = None if len(boxes) == 0 else boxes

                wandb_image = wandb.Image(
                    image,
                    masks=masks,
                    boxes=boxes,
                    classes=[
                        {"id": c, "name": name}
                        for c, name in (
                            point_sample_categories | cur_sample_categories
                        ).items()
                    ],
                )

                self.add_image_to_sequence(
                    sequence_name,
                    f"image_{image_idx}_sample_{i}_substep_{substitution_step}_prompts",
                    wandb_image,
                    metadata=[epoch, dataset_names[i]],
                )

    @main_process_only
    def create_image_sequence(self, name, columns=[]):
        self.sequences[name] = wandb.Table(["ID", "Image"] + columns)

    @main_process_only
    def add_image_to_sequence(
        self, sequence_name, name, wandb_image: wandb.Image, metadata=[]
    ):
        self.sequences[sequence_name].add_data(name, wandb_image, *metadata)

    @main_process_only
    def add_image_sequence(self, name):
        wandb.log({f"{self.context}_{name}": self.sequences[name]})
        del self.sequences[name]

    @main_process_only
    def log_asset_folder(self, folder, base_path=None, step=None):
        files = os.listdir(folder)
        for file in files:
            wandb.save(os.path.join(folder, file), base_path=base_path)

    @main_process_only
    def log_metric(self, name, metric, epoch=None):
        wandb.log({f"{self.context}/{name}": metric})

    @main_process_only
    def log_metrics(self, metrics: dict, epoch=None):
        wandb.log({f"{self.context}/{k}": v for k, v in metrics.items()})

    def log_training_state(self, epoch, subfolder):
        logger.info("Waiting for all processes to finish for saving training state")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process:
            subdir = os.path.join(self.local_dir, subfolder)
            self.accelerator.save_state(output_dir=subdir)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process:
            self.log_asset_folder(subdir, step=epoch, base_path=self.local_dir)

    def __repr__(self):
        return "WandbLogger"

    @contextmanager
    def train(self):
        # Save the old context and set the new one
        old_context = self.context
        self.context = "train"

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def validate(self):
        # Save the old context and set the new one
        old_context = self.context
        self.context = "validate"

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def test(self):
        # Save the old context and set the new one
        old_context = self.context
        self.context = "test"

        yield self

        # Restore the old one
        self.context = old_context

    @property
    def name(self):
        if self.accelerator.is_local_main_process:
            return self.experiment.name
        return None

    @property
    def url(self):
        if self.accelerator.is_local_main_process:
            return self.experiment.url
        return None

