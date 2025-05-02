import logging
import colorlog


def get_logger(name, log_file=None):
    """
    Creates a logger with both StreamHandler (colored output to console) and FileHandler (plain output to a log file).

    :param name: Name of the logger (used in log messages)
    :param log_file: Path to the log file
    :return: Configured logger instance
    """
    name = name.split('.')[-1]

    # Create a StreamHandler with colorlog formatter
    stream_handler = colorlog.StreamHandler()
    stream_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-2s%(reset)s %(blue)s%(asctime)s%(reset)s %(purple)s[%(name)s]%(reset)s %(message)s",
        datefmt='[%m-%d %H:%M:%S]',
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        style='%'
    )
    stream_handler.setFormatter(stream_formatter)

    if log_file is not None:
        # Create a FileHandler with a plain text formatter
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(levelname)-2s %(asctime)s [%(name)s] %(message)s",
            datefmt='[%m-%d %H:%M:%S]'
        )
        file_handler.setFormatter(file_formatter)

    # Create a logger with the specified name
    logger = colorlog.getLogger(name)

    # Check if the logger already has handlers, if yes, clear them
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set logging level to DEBUG
    logger.setLevel(logging.DEBUG)

    # Add both handlers to the logger
    logger.addHandler(stream_handler)
    if log_file is not None:
        logger.addHandler(file_handler)

    return logger
