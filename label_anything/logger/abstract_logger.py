from label_anything.logger.text_logger import get_logger
from label_anything.utils.utils import log_every_n
import os
import time
from PIL import Image
from accelerate import Accelerator


logger = get_logger(__name__)


class AbstractLogger:
    def __init__(
        self,
        experiment,
        accelerator: Accelerator,
        tmp_dir: str,
        log_frequency: int = 100,
        train_image_log_frequency: int = 1000,
        val_image_log_frequency: int = 1000,
        test_image_log_frequency: int = 1000,
        experiment_save_delta: int = None,
    ):
        self.experiment = experiment
        self.accelerator = accelerator
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.log_frequency = log_frequency
        self.prefix_frequency_dict = {
            "train": train_image_log_frequency,
            "val": val_image_log_frequency,
            "test": test_image_log_frequency,
        }
        self.start_time = time.time()
        self.experiment_save_delta = experiment_save_delta

    def _get_class_ids(self, classes):
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
            )

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
        pass

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
        pass

    def log_image(
        self, name: str, image_data: Image, annotations=None, metadata=None, step=None
    ):
        pass

    def add_tags(self, tags):
        pass
        
    def log_parameters(self, params):
        pass
    
    def log_metric(self, name, metric, epoch=None):
        pass

    def log_metrics(self, metrics, epoch=None):
        pass

    def log_parameter(self, name, parameter):
        pass

    def log_training_state(self, epoch):
        pass

    def save_experiment_timed(self):
        """
        Save the experiment every `self.time_delta` seconds
        """
        pass

    def save_experiment(self):
        pass
    
    def log_asset_folder(self, path):
        pass
    
    def train(self):
        pass
    
    def validate(self):
        pass
        
    def test(self):
        pass

    def end(self):
        pass
    
    @property
    def name(self):
        return self.experiment.name

    @property
    def url(self):
        return self.experiment.url


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
