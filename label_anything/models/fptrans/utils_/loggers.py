import logging
import os
from datetime import datetime
from time import time, strftime, localtime

from label_anything.models.fptrans.constants import on_cloud


levels = [logging.NOTSET,
          logging.DEBUG,
          logging.INFO,
          logging.WARNING,
          logging.ERROR,
          logging.CRITICAL]


class Logger(object):
    _log = logging.getLogger()
    default = True

    def debug(self, *args, **kwargs):
        self._log.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self._log.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self._log.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self._log.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        self._log.critical(*args, **kwargs)


# We create a global logger. This statement is only executed once.
logger = Logger()


class MyFormatter(logging.Formatter):
    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.strftime("%Y-%m-%d %H:%M:%S")
            # s = "%s.%03d" % (t, record.msecs)
        return s


def create_logger(log_file=None, file_=False, console=True,
                  with_time=False, file_level=2, console_level=2,
                  propagate=False, clear_exist_handlers=False, name=None):
    """ Create a logger to write info to console and file.
    
    Params
    ------
    `log_file`: string, path to the logging file  
    `file_`: write info to file or not  
    `console`: write info to console or not  
    `with_time`: if set, log_file will be add a time prefix
    `file_level`: file info level  
    `console_level`: console info level  
    `propagate`: if set, then message will be propagate to root logger  
    `name`: logger name, if None, then root logger will be used  

    Note:
    * don't set propagate flag and give a name to logger is the way
    to avoid logging dublication.
    * use code snippet below to change the end mark "\n"
    ```
    for hdr in logger.handlers:
        hdr.terminator = ""
    ```
    
    Returns
    -------
    A logger object of class getLogger()
    """
    if file_:
        prefix = strftime('%Y%m%d%H%M%S', localtime(time()))
        if log_file is None:
            log_file = os.path.join(os.path.dirname(__file__), prefix)
        elif with_time:
            log_file = os.path.join(os.path.dirname(log_file), prefix + "_" + os.path.basename(log_file))
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)

    if clear_exist_handlers:
        logger.handlers.clear()

    logger.setLevel(levels[1])
    logger.propagate = propagate

    formatter = MyFormatter("%(asctime)s: %(levelname).1s %(message)s")

    if file_:
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(levels[file_level])
        file_handler.setFormatter(formatter)
        # Register handler
        logger.addHandler(file_handler)

    if console:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(levels[console_level])
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_global_logger(name="default"):
    global logger
    if logger.default:
        logger._log = create_logger(name=name)
        logger.default = False
    return logger


class C(object):
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def c(string, color):
        """ Change color of string """
        if on_cloud:
            return string
        return color + string + C.ENDC


def test_Color():
    print(C.c("Header", C.HEADER))
    print("Processing ...", C.c("OK", C.OKBLUE))
    print("Processing ...", C.c("OK", C.OKGREEN))
    print(C.c("Warning", C.WARNING))
    print(C.c("Failed", C.FAIL))
    print(C.c("Bold", C.BOLD))
    print(C.c("Underline", C.UNDERLINE))
