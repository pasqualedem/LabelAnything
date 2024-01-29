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
from label_anything.logger.abstract_logger import AbstractLogger

from label_anything.logger.text_logger import get_logger
from accelerate import Accelerator
from label_anything.logger.utils import get_tmp_dir, take_image
from label_anything.utils.utils import log_every_n

from label_anything.visualization.visualize import get_image


logger = get_logger(__name__)

WANDB_ID_PREFIX = "wandb_id."
WANDB_INCLUDE_FILE_NAME = ".wandbinclude"


def wandb_experiment(accelerator: Accelerator, params: dict):
    logger_params = deepcopy(params.get("logger", {}))
    wandb_params = logger_params.pop("wandb", {})
    wandb_information = {
        "accelerator": accelerator,
        "project_name": params["experiment"]["name"],
        **wandb_params,
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

    if wandb_information.pop("offline"):
        os.environ["WANDB_MODE"] = "offline"

    wandb_logger = WandBLogger(**wandb_information, **logger_params)
    wandb_logger.log_parameters(params)
    wandb_logger.add_tags(logger_params.get("tags", ()))

    return wandb_logger


class WandBLogger(AbstractLogger):
    MAX_CLASSES = 100000  # For negative classes

    def __init__(
        self,
        project_name: str,
        resumed: bool = False,
        offline_directory: str = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = True,
        save_logs_remote: bool = True,
        entity: Optional[str] = None,
        api_server: Optional[str] = None,
        save_code: bool = False,
        tags=None,
        run_id=None,
        **kwargs,
    ):
        """

        :param experiment_name: Used for logging and loading purposes
        :param s3_path: If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param checkpoint_loaded: if true, then old tensorboard files will *not* be deleted when tb_files_user_prompt=True
        :param max_epochs: the number of epochs planned for this training
        :param tb_files_user_prompt: Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard: Whether to launch a TensorBoard process.
        :param tensorboard_port: Specific port number for the tensorboard to use when launched (when set to None, some free port
                    number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote: Saves log files in s3.
        :param save_code: save current code to wandb
        """
        self.resumed = resumed
        resume = "must" if resumed else None
        if offline_directory:
            os.makedirs(offline_directory, exist_ok=True)
        experiment = wandb.init(
            project=project_name,
            entity=entity,
            resume=resume,
            id=run_id,
            tags=tags,
            dir=offline_directory,
            group=kwargs.get("group"),
        )
        super().__init__(experiment=experiment, **kwargs)
        if save_code:
            self._save_code()

        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        self.save_logs_wandb = save_logs_remote
        self.context = ""
        self.sequences = {}
        
        wandb.define_metric("train/step")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="train/step")
        
        wandb.define_metric("validate/step")
        # set all other validate/ metrics to use this step
        wandb.define_metric("validate/*", step_metric="train/step")

        # self._set_wandb_id(self.experiment.id)
        if api_server is not None:
            if api_server != os.getenv("WANDB_BASE_URL"):
                logger.warning(
                    f"WANDB_BASE_URL environment parameter not set to {api_server}. Setting the parameter"
                )
                os.putenv("WANDB_BASE_URL", api_server)

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

    def log_parameters(self, config: dict = None):
        wandb.config.update(config, allow_val_change=self.resumed)

    def add_tags(self, tags):
        wandb.run.tags = wandb.run.tags + tuple(tags)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        wandb.log(data={tag: scalar_value}, step=global_step)

    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        for name, value in tag_scalar_dict.items():
            if isinstance(value, dict):
                tag_scalar_dict[name] = value["value"]
        wandb.log(data=tag_scalar_dict, step=global_step)

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

    def add_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.array],
        bins: str,
        global_step: int = 0,
    ):
        wandb.log({tag: wandb.Histogram(values, num_bins=bins)}, step=global_step)

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

    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        wandb.log({tag: text_string}, step=global_step)

    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        wandb.log({tag: figure}, step=global_step)

    def add_mask(self, tag: str, image, mask_dict, global_step: int = 0):
        wandb.log({tag: wandb.Image(image, masks=mask_dict)}, step=global_step)

    def add_table(self, tag, data, columns, rows):
        if isinstance(data, torch.Tensor):
            data = [[x.item() for x in row] for row in data]
        table = wandb.Table(data=data, rows=rows, columns=columns)
        wandb.log({tag: table})

    def end(self):
        wandb.finish()

    def add_file(self, file_name: str = None):
        wandb.save(
            glob_str=os.path.join(self._local_dir, file_name),
            base_path=self._local_dir,
            policy="now",
        )

    def add_summary(self, metrics: dict):
        wandb.summary.update(metrics)

    def upload(self):
        if self.save_tensorboard_wandb:
            wandb.save(
                glob_str=self._get_tensorboard_file_name(),
                base_path=self._local_dir,
                policy="now",
            )

        if self.save_logs_wandb:
            wandb.save(
                glob_str=self.log_file_path, base_path=self._local_dir, policy="now"
            )

    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"

        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints_wandb:
            if self.s3_location_available:
                self.model_checkpoints_data_interface.save_remote_checkpoints_file(
                    self.experiment_name, self._local_dir, name
                )
            wandb.save(glob_str=path, base_path=self._local_dir, policy="now")

    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError as e:
            logger.warning("tensorboard file could not be located for ")
            return None

        return tb_file_path

    def _get_wandb_id(self):
        for file in os.listdir(self._local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                return file.replace(WANDB_ID_PREFIX, "")

    def _set_wandb_id(self, id):
        for file in os.listdir(self._local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                os.remove(os.path.join(self._local_dir, file))

    def add(self, tag: str, obj: Any, global_step: int = None):
        pass

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

    def log_batch(
        self,
        batch_idx,
        image_idx,
        batch_size,
        epoch,
        step,
        substitution_step,
        input_dict,
        gt,
        pred,
        dataset,
        dataset_names,
        phase,
    ):
        if log_every_n(image_idx, self.prefix_frequency_dict[phase]):
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
            sequence_name = "predictions"
            self.create_image_sequence(sequence_name, columns=["Epoch", "Dataset"])
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
                images=query_images,
                gt=gt,
                pred=pred,
                categories=categories,
                dataset_names=dataset_names,
                prefix=phase,
                sequence_name=sequence_name,
            )
            self.add_image_sequence(sequence_name)

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
        sequence_name,
    ):
        dims = input_dict["dims"]
        classes = self._get_class_ids(input_dict["classes"])

        for b in range(gt.shape[0]):
            image = get_image(take_image(images[b], dims[b, 0]))
            cur_dataset_categories = categories[dataset_names[b]]
            cur_sample_categories = {
                k + 1: cur_dataset_categories[c]["name"]
                for k, c in enumerate(classes[b])
            }
            cur_sample_categories[0] = "background"

            sample_gt = gt[b, : dims[b, 0, 0], : dims[b, 0, 1]].detach().cpu().numpy()

            sample_pred = pred[b, :, : dims[b, 0, 0], : dims[b, 0, 1]]
            sample_pred = torch.argmax(sample_pred, dim=0).detach().cpu().numpy()

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
                metadata=[epoch, dataset_names[b]]
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
        sequence_name,
    ):
        all_masks = (
            input_dict["prompt_masks"].argmax(dim=2)
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
                k + 1: cur_dataset_categories[c]["name"]
                for k, c in enumerate(classes[i])
            }
            cur_sample_categories[0] = "background"
            mask_sample_categories = deepcopy(cur_sample_categories)
            sample_images = images[i]
            for j in range(all_masks.shape[1]):
                image = get_image(sample_images[j])
                # log masks, boxes and points
                for c in range(1, input_dict["prompt_masks"].shape[2]):
                    if c > len(classes[i]):
                        break
                    boxes = all_boxes[i, j, c]
                    points = all_points[i, j, c]
                    flag_boxes = flags_boxes[i, j, c]
                    flag_points = flags_points[i, j, c]
                    label = cur_sample_categories[c]

                    box_data = []
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

                    points_data = []
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
                                cur_sample_categories[class_id] = point_label

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
                if flags_masks[i, j].sum() > 0:
                    cur_mask = all_masks[i, j].unsqueeze(0).unsqueeze(0).float()
                    masks = {
                        "prompts": {
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
                if len(box_data) > 0:
                    boxes["boxes"] = {
                        "box_data": box_data,
                        "class_labels": cur_sample_categories,
                    }
                if len(points_data) > 0:
                    boxes["points"] = {
                        "box_data": points_data,
                        "class_labels": cur_sample_categories,
                    }
                boxes = None if len(boxes) == 0 else boxes

                wandb_image = wandb.Image(
                    image,
                    masks=masks, 
                    boxes=boxes,
                    classes=[
                        {"id": c, "name": name}
                        for c, name in cur_sample_categories.items()
                    ],
                )

                self.add_image_to_sequence(
                    sequence_name,
                    f"image_{image_idx}_sample_{i}_substep_{substitution_step}_prompts",
                    wandb_image,
                    metadata=[epoch, dataset_names[i]]
                )

    def create_image_sequence(self, name, columns=[]):
        self.sequences[name] = wandb.Table(["ID", "Image"] + columns)

    def add_image_to_sequence(self, sequence_name, name, wandb_image: wandb.Image, metadata=[]):
        self.sequences[sequence_name].add_data(name, wandb_image, *metadata)

    def add_image_sequence(self, name):
        wandb.log({f"{self.context}_{name}": self.sequences[name]})
        del self.sequences[name]

    def log_asset_folder(self, folder, step=None):
        files = os.listdir(folder)
        for file in files:
            wandb.log_artifact(os.path.join(folder, file), step=step)

    def log_metric(self, name, metric, epoch=None):
        wandb.log({f"{self.context}/{name}": metric})

    def log_metrics(self, metrics: dict, epoch=None):
        wandb.log({f"{self.context}/{k}": v for k, v in metrics.items()})

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
