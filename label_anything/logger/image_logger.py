import torch
import comet_ml
import os
from PIL import Image


class Logger(comet_ml.CometLogger):
    def __init__(self, project_name, api_key, experiment_name):
        super().__init__(project_name, api_key, experiment_name)

    def log_image_input(self, image, bboxes, points, mask):
        image = Image.fromarray(image.cpu().numpy())
        self.log_image(image, name="image_input")
        self.log_tensor(bboxes, name="bboxes")
        self.log_tensor(points, name="points")
        self.log_tensor(mask, name="mask")

    def log_segmentation_output(self, segmentation):
        self.log_tensor(segmentation, name="segmentation")


if __name__ == "__main__":
    logger = Logger(
        project_name="segmentation",
        api_key=os.getenv("COMET_API_KEY"),
        experiment_name="segmentation",
    )
    # logger.log_image_input(torch.rand(1, 3, 256, 256), torch.rand(1, 4), torch.rand(1, 2), torch.rand(1, 256, 256))
    # logger.log_segmentation_output(torch.rand(1, 256, 256))
