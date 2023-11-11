from label_anything.visualization.visualize import get_image
from label_anything.logger.utils import (
    extract_polygons_from_tensor,
    crop_padding,
    resize_gt,
    take_image,
    extract_masks_dynamic,
)
import math
import torch.nn.functional as F


class Logger:
    def __init__(self, experiment):
        self.experiment = experiment

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

    def log_pred(self, batch_idx, input_dict, pred):
        images = input_dict["images"]
        data = []
        annotations = [{"name": "Prediction", "data": data}]
        for b in pred.shape[0]:
            sample_images = images[b]
            image = get_image(sample_images)
            mask = pred[b]
            polygons = extract_polygons_from_tensor(mask)
            data.append({"points": polygons, "score": None})

        self.experiment.log_image(
            image_data=image,
            annotations=annotations,
            metadata={"batch_idx": batch_idx},
        )

    def log_gt(self, batch_idx, step, input_dict, gt, categories):
        images = input_dict["images"]
        dims = input_dict["dims"]
        data = []
        annotations = [{"name": "Ground truth", "data": data}]

        for b in range(gt.shape[0]):
            n_gt = resize_gt(crop_padding(gt[b, 0]).float(), dims[b, 0])
            masks = extract_masks_dynamic(n_gt)
            sample_images = images[b]
            image = get_image(sample_images)
            for i, mask in enumerate(masks):
                polygons = extract_polygons_from_tensor(mask)
                data.append({"points": polygons, "score": None})

                self.experiment.log_image(
                    image_data=image,
                    annotations=annotations,
                    metadata={
                        "batch_idx": batch_idx,
                        "gt_idx": i,
                        "substitution_step": step,
                    },
                )

    def log_batch(self, batch_idx, step, input_dict, categories):
        images = input_dict["images"]
        all_masks = input_dict["prompt_masks"]
        all_boxes = input_dict["prompt_bboxes"]
        all_points = input_dict["prompt_points"]
        flags_masks = input_dict["flag_masks"]
        flags_boxes = input_dict["flag_bboxes"]
        flags_points = input_dict["flag_points"]
        classes = self.__get_class_ids(input_dict["classes"])
        print(classes)

        for i in range(images.shape[0]):
            sample_images = images[i]
            for j in range(sample_images.shape[0]):
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
                    label = categories[classes[i][c - 1]]["name"]

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

                    points_log = []
                    for k in range(points.shape[0]):
                        if flag_points[k] == 1:
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
                            points_log.append(ps)
                    if len(points_log) > 0:
                        # print(points_log)
                        data.append(
                            {"points": points_log, "label": label, "score": None}
                        )
                print("log image")
                self.experiment.log_image(
                    image_data=image,
                    annotations=annotations,
                    metadata={
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "substitution_step": step,
                    },
                )

    def log_image(self, img_data, annotations=None):
        self.experiment.log_image(
            image_data=img_data,
            annotations=annotations,
        )

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
