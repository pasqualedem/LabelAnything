class Logger:
    def __init__(self, experiment):
        self.experiment = experiment

    def log_image(self, img_data, annotations=None):
        self.experiment.log_image(
            image_data=img_data,
            annotations=annotations,
        )

    def log_metric(self, nome, metric):
        self.experiment.log_metric(
            nome,
            metric,
        )

    def log_parameter(self, nome, parameter):
        self.experiment.log_parameter(
            nome,
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
