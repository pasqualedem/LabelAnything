# width x height x channels
class Logger:
    def __init__(self, experiment):
        self.experiment = experiment

    def log_image(self, img_data, annotation):
        self.experiment.log_image(img_data, annotation)

    def log_artifact(self, nome, annotation):
        self.experiment.log_artifact(nome, annotation)

    def log_metric(self, nome, metric):
        self.experiment.log_metric(nome, metric)

    def _tensor_to_mask(self, tensor):
        return mask

    def _tensor_to_bboxes(self, tensor):
        return bbox

    def _create_bbox_annotation(data: list[dict]):
        annotation = [
            {
                "name": "predictions",
                "data": data,
            }
        ]
        return annotation

    def _create_mask_annotation(data: list[dict]):
        annotation = [
            {
                "name": "predictions",
                "data": data,
            }
        ]
        return annotation


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
annotation = {"bounding_box": [0, 0, 100, 100]}
annotation = [
    {
        "class": 'ciola',
        "bounding_box": [0, 0, 100, 100],
    },
    {
        "class": 'figa'
        "bounding_box": [0, 0, 100, 101],
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
