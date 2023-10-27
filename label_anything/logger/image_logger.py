class Logger:
    def __init__(self, experiment):
        self.experiment = experiment

    def log_image(self, img_data, annotations=None):
        self.experiment.log_image(
            image_data=img_data,
            annotations=annotations,
        )

    def log_metric(self, name, metric, epoch=None):
        self.experiment.log_metric(
            name,
            metric,
            epoch
        )

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
