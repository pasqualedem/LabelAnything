from logger.text_logger import get_logger


logger = get_logger(__name__)


class Logger:
    def __init__(self, experiment):
        self.experiment = experiment

    def log_image(self, img_data, annotations):
        self.experiment.log_image(
            image_data=img_data,
            annotations=annotations,
        )

    def log_artifact(self, nome, annotation):
        self.experiment.log_artifact(nome, annotation)

    def log_metric(self, nome, metric):
        self.experiment.log_metric(nome, metric)

    def structure_annotations(annotations):
        return [{"name": "Predictions", "data": annotations}]

    def __data_to_single__(data: dict):
        output = {}
        for key in data:
            output[key] = data[key]

        # Estrae i tensori dal dizionario output
        (
            query_image,
            example_images,
            point_coords,
            point_labels,
            boxes,
            mask_inputs,
            box_flags,
        ) = output.values()

        return (
            query_image,
            example_images,
            point_coords,
            point_labels,
            boxes,
            mask_inputs,
            box_flags,
        )


"""
'query_image': The input image as a torch tensor in Bx3xHxW format,
                already transformed for input to the model.
'example_images': The example images as a torch tensor in BxMx3xHxW format,
                already transformed for input to the model.
'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxMxCxNx2. Already transformed to the
                input frame of the model.
'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxMxCxN.
'boxes': (torch.Tensor) Batched box inputs, with shape BxMxCxNx4.
                Already transformed to the input frame of the model.
'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form BxMxCxHxW.
'box_flags': (torch.Tensor) Batched bounding box flags, with shape BxMxCxN.
"""


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
annotation = {"boxes": [x, y, w, h]}
annotation = [
    {
        "label": 'ciola',
        "boxes": [0, 0, 100, 100],
    },
    {
        "label": 'figa'
        "boxes": [0, 0, 100, 101],
    }
]
$ logger.log_artifact("annotation_name", annotation)
==============================================================
punto = {
    "x": 100,
    "y": 200,
}
punti = [
    {
        "x": 100,
        "y": 200,
    },
    {
        "x": 200,
        "y": 300,
    },
]
$ logger.log_artifact("punti_name", punto)
==============================================================
maschera = {
    "points": [
        [100, 200],
        [200, 300],
        [300, 200],
        [200, 100],
    ],
}
maschere = [
    {
        "points": [
            [100, 200],
            [200, 300],
            [300, 200],
            [200, 100],
        ],
    },
    {
        "points": [
            [400, 500],
            [500, 600],
            [600, 500],
            [500, 400],
        ],
    },
]
$ logger.log_artifact("maschera", maschera)
==============================================================
"""
