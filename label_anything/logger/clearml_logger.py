import os
import shutil
from typing import Optional, Union
from subprocess import check_output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from PIL import Image
from flatbuffers.builder import np
from matplotlib import pyplot as plt
from clearml import Task, OutputModel, Model
from super_gradients.common.environment.ddp_utils import multi_process_safe

from utils.segmentation import tensor_to_segmentation_image
from logger.text_logger import get_logger
from logger.basesg_logger import AbstractRunWrapper, BaseSGLogger

logger = get_logger(__name__)

WANDB_ID_PREFIX = "wandb_id."
WANDB_INCLUDE_FILE_NAME = ".wandbinclude"


class ClearMLLogger(BaseSGLogger):
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        storage_location: str,
        resumed: bool,
        training_params: dict,
        checkpoints_dir_path: str,
        tb_files_user_prompt: bool = False,
        launch_tensorboard: bool = False,
        tensorboard_port: int = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = True,
        run: Task = None,
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
        self.s3_location_available = storage_location.startswith("s3")
        self.resumed = resumed
        resume = "must" if resumed else None
        if kwargs.get("group"):
            project_name = f"{project_name}/{kwargs.get('group')}"
        if not experiment_name:
            experiment_name = (
                check_output(["adjectiveanimalnumber"])
                .decode()
                .rstrip("\n")
                .rstrip("\r")
            )

        if run is not None:
            self.run = run.clearml_task
            experiment_name = run.clearml_task.name
        else:
            self.run = Task.init(
                project_name=project_name,
                task_name=experiment_name,
                auto_connect_frameworks=False,
                continue_last_task=resume,
                output_uri=True,
            )
        if save_code:
            self._save_code()

        checkpoints_dir_path = f"{checkpoints_dir_path}/{experiment_name}"

        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        super().__init__(
            project_name,
            experiment_name,
            storage_location,
            resumed,
            training_params,
            checkpoints_dir_path,
            tb_files_user_prompt,
            launch_tensorboard,
            tensorboard_port,
            self.s3_location_available,
            self.s3_location_available,
            self.s3_location_available,
        )

    @multi_process_safe
    def add_config(self, tag: str = None, config: dict = None):
        self.run.connect(fix_clearml_empty_map(config), name=tag)

    @multi_process_safe
    def add_scalar(
        self, tag: str, scalar_value: float, global_step: int = 0, series=None
    ):
        if series is None:
            series = tag
        self.run.get_logger().report_scalar(
            title=series, series=tag, value=scalar_value, iteration=global_step
        )

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        for name, value in tag_scalar_dict.items():
            if isinstance(value, dict):
                self.add_scalar(
                    name,
                    value["value"],
                    global_step=global_step,
                    series=value["series"],
                )
            else:
                self.add_scalar(name, value, global_step=global_step)

    @multi_process_safe
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
        self.run.get_logger().report_image(
            title=tag, series=tag, image=image, iteration=global_step
        )

    @multi_process_safe
    def add_images(
        self,
        tag: str,
        images: Union[torch.Tensor, np.array],
        data_format="NCHW",
        global_step: int = 0,
    ):
        raise NotImplementedError()

    @multi_process_safe
    def add_video(
        self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0
    ):
        raise NotImplementedError()

    @multi_process_safe
    def add_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.array],
        bins: str,
        global_step: int = 0,
    ):
        self.run.get_logger().report_histogram(
            title=tag, series=tag, values=values, iteration=global_step
        )

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        self.run.get_logger().report_text(msg=text_string)

    @multi_process_safe
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        self.run.get_logger().report_matplotlib_figure(
            title=tag, figure=figure, iteration=global_step
        )

    @multi_process_safe
    def add_table(self, tag, data, columns, rows):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data=data, columns=columns)
        self.run.get_logger().report_table(title=tag, series=tag, table_plot=data)

    @multi_process_safe
    def add_plot(
        self, tag: str, values: pd.DataFrame, xtitle, ytitle, classes_marker=None
    ):
        if classes_marker:
            for cls in values[classes_marker].unique():
                scatter2d = (
                    values[values[classes_marker] == cls]
                    .loc[:, values.columns != classes_marker]
                    .values
                )

                self.run.get_logger().current_logger().report_scatter2d(
                    tag,
                    tag,
                    iteration=None,
                    scatter=scatter2d,
                    xaxis=xtitle,
                    yaxis=ytitle,
                    mode="lines",
                )
        else:
            self.run.get_logger().current_logger().report_scatter2d(
                tag,
                tag,
                iteration=None,
                scatter=values.values,
                xaxis=xtitle,
                yaxis=ytitle,
                mode="lines+markers",
            )

    @multi_process_safe
    def close(self, really=False, failed=False):
        if really:
            super().close()
            OutputModel.wait_for_uploads()
            if failed:
                self.run.mark_failed()
            else:
                shutil.rmtree(self._local_dir)
                self.run.close()

    @multi_process_safe
    def add_file(self, file_name: str = None):
        self.run.upload_artifact(
            file_name, artifact_object=os.path.join(self._local_dir, file_name)
        )

    @multi_process_safe
    def add_summary(self, metrics: dict):
        for name, value in metrics.items():
            self.run.get_logger().report_single_value(name=name, value=value)

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = "ckpt.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"
        models = [model.name for model in self.run.models["output"]]
        if name in models:
            model = self.run.models["output"][models.index(name)]
            OutputModel.wait_for_uploads()
            Model.remove(model)
        model = OutputModel(task=self.run, name=name)
        path = os.path.join(self._local_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        model.update_weights(
            weights_filename=path, iteration=global_step, auto_delete_file=False
        )

    @multi_process_safe
    def add_mask(self, tag: str, image, mask_dict, global_step: int = 0):
        cmap = mask_dict["predictions"]["cmap"]
        labels = mask_dict["ground_truth"]["id2label"]
        predictions = tensor_to_segmentation_image(
            mask_dict["predictions"]["mask_data"],
            labels=mask_dict["predictions"]["id2label"],
            cmap=cmap,
        )

        ground_truth, clmap = tensor_to_segmentation_image(
            mask_dict["ground_truth"]["mask_data"],
            cmap=cmap,
            labels=mask_dict["ground_truth"]["id2label"],
            return_clmap=True,
        )
        clmap = {
            labels[i]: "#%02x%02x%02x"
            % (clmap[labels[i]][0], clmap[labels[i]][1], clmap[labels[i]][2])
            for i in range(len(labels))
        }
        data = np.stack([image, predictions, ground_truth])
        fig = px.imshow(data, facet_col=0, title=tag)
        annotations = ["image", "predictions", "ground_truth"]
        for k in range(len(annotations)):
            fig.layout.annotations[k].update(text=annotations[k])
        fig.add_traces(
            [
                go.Scatter(
                    x=[None],
                    y=[None],
                    name=name,
                    mode="markers",
                    marker=dict(color=color, size=1),
                )
                for name, color in clmap.items()
            ]
        )
        self.run.get_logger().report_plotly(
            title=tag, series=tag, iteration=global_step, figure=fig
        )

    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError as e:
            logger.warning("tensorboard file could not be located for ")
            return None

        return tb_file_path

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

    def create_image_mask_sequence(self, name):
        pass

    def add_image_mask_to_sequence(self, sequence_name, name, image, mask_dict):
        self.add_mask(name, image, mask_dict)

    def add_image_mask_sequence(self, name):
        pass

    def add_plotly_figure(self, tag, figure, global_step=None):
        self.run.get_logger().report_plotly(
            title=tag, series=tag, iteration=global_step, figure=figure
        )

    @property
    def name(self):
        return self.run.name

    @property
    def url(self):
        return self.run.get_output_log_web_page()

    def __repr__(self):
        return "ClearMLLogger"

    @classmethod
    def get_interrupted_run(cls, input_settings):
        namespace = input_settings["name"]
        group = input_settings["group"]
        project = namespace + "/" + group
        last_run = Task.get_task(project_name=project)
        stage = ["train", "test"]
        updated_config = None
        return ClearmlRunWrapper(last_run), updated_config, stage


class ClearmlRunWrapper(AbstractRunWrapper):
    def __init__(self, clearml_task: Task) -> None:
        super().__init__()
        clearml_task.mark_started(force=True)
        self.clearml_task = clearml_task

    def get_params(self):
        return restore_clearml_empty_map(
            self.clearml_task.get_parameters_as_dict().get("General").get("in_params")
        )

    def update_params(self, params):
        conf = self.clearml_task.get_parameters_as_dict()
        conf["General"]["in_params"] = params
        self.clearml_task.set_parameters_as_dict(conf)

    def get_summary(self):
        return self.clearml_task.get_reported_single_values()

    @property
    def id(self):
        return self.clearml_task.id

    @property
    def group(self):
        return self.clearml_task.get_project_name().split("/")[0]

    def get_local_checkpoint_path(self, phases):
        track_dir = (
            self.get_params().get("experiment").get("tracking_dir") or "experiments"
        )
        checkpoint_path_group = os.path.join(
            track_dir, self.clearml_task.get_project_name(), self.clearml_task.name
        )
        checkpoint_path = None
        if "epoch" in self.get_summary():
            ckpt = "ckpt_latest.pth" if "train" in phases else "ckpt_best.pth"
            if os.path.exists(os.path.join(checkpoint_path_group, ckpt)):
                checkpoint_path = os.path.join(checkpoint_path_group, ckpt)
            else:
                logger.error(
                    f"{self.clearml_task.name} not found in {checkpoint_path_group}"
                )
                raise ValueError(
                    f"{self.clearml_task.name} not found in {checkpoint_path_group}"
                )
        return checkpoint_path


def load_weight_from_clearml(task_name, model_name="ckpt_best"):
    t = Task.get_task(task_name=task_name)
    model = t.models["output"][model_name]
    path = model.get_weights()
    return torch.load(path)


def fix_clearml_empty_map(dictionary):
    if dictionary == {}:
        return "__empty__"
    for k, v in dictionary.items():
        if isinstance(v, dict):
            dictionary[k] = fix_clearml_empty_map(v)
    return dictionary


def restore_clearml_empty_map(dictionary):
    for k, v in dictionary.items():
        if isinstance(v, dict):
            dictionary[k] = restore_clearml_empty_map(v)
        elif v == "__empty__":
            dictionary[k] = {}
    return dictionary


def upload_to_clearml(task_name, project_name, files):
    t = Task.get_task(task_name=task_name)
    if t is None:
        t = Task.create(task_name=task_name, project_name=project_name)
    # add and upload a wildcard
    for file in files:
        t.upload_artifact(file, artifact_object=file)
