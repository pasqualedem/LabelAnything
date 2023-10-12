import comet_ml
from label_anything.logger.image_logger import Logger


# set configuration
def comet_experiment(comet_information, hyper_params):
    comet_ml.init(comet_information)
    experiment = comet_ml.Experiment()
    experiment.add_tags(hyper_params["tags"])
    experiment.log_parameters(hyper_params)
    logger = Logger(experiment)
    return logger, experiment
