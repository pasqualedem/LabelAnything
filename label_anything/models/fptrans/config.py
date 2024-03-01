import os
import sys
import random
import shutil
from pathlib import Path
import tarfile

import numpy as np
import torch
from torch.backends import cudnn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.config.custom_containers import ReadOnlyDict
from sacred.observers import FileStorageObserver, MongoObserver

from label_anything.models.fptrans.constants import on_cloud, output_dir
from utils_.loggers import get_global_logger

SETTINGS.DISCOVER_SOURCES = "sys"
SETTINGS.DISCOVER_DEPENDENCIES = "sys"


def setup(ex):
    # Track outputs
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    @ex.config
    def ex_config():
        # Global configurations ===========================================================
        log_dir = "output"              # str, Directory to save logs, model parameters, etc
        fileStorage = True              # bool, enable fileStorage observer
        mongodb = False                 # bool, enable MongoDB observer
        mongo_host = 'localhost'        # str, MongoDB host
        mongo_port = 7000               # int, MongoDB port

        print_interval = 200            # print interval, by iteration
        tqdm = False                    # bool, only enable tqdm in an interactive terminal
        if not on_cloud:
            tqdm = True

        shot = 1                        # int, number of support samples per episode
        split = 0                       # int, split number [0, 1, 2, 3], required
        seed = 1234                     # int, training set random seed, fixed for reproducibility
        test_seed = 5678                # int, test set random seed, fixed for reproducibility

        # Checkpoint configurations =========================================================
        ckpt_interval = 1               # int, checkpoint interval, 0 to disable checkpoint
        ckpt = ''                       # str, checkpoint file
        exp_id = -1                     # experiment id to load checkpoint. -1 means `ckpt` is full path.
        no_resume = False               # bool, set to True if you don't want to load weights during testing
        strict = True                   # bool, load weights in strict mode, see model.load_state_dict()

        # Data configurations ==============================================================
        proc = 'cv2'                    # str, processor, [cv2|pil], 'cv2' is required here
        height = 480
        width = 480
        scale_min = 0.9
        scale_max = 1.1
        dataset = "PASCAL"              # str, dataset name. [PASCAL|COCO]
        use_split_coco = True
        rotate = 10
        pad_type = 'reflect'            # str, pad type, reflect or constant

        bs = 4                          # int, batch size
        test_bs = 1                     # int, test batch size (Don't change it!)
        num_workers = min(bs, 16)       # int, PyTorch DataLoader argument
        train_n = 0                     # int, number of train examples in each epoch (for balancing dataset)
        test_n = 5000                   # int, number of test examples in each run
        test_range = (None, None)       # tuple, test range from ... to ...
        coco2pascal = False             # bool, flag for evaluating in domain shift scenario of coco -> pascal

        # Training configurations  ============================================================
        epochs = 60                     # int, Number of total epochs for training
        lr = 0.001                      # float, Base learning rate for model training
        lrp = "period_step"             # str, Learning rate policy [custom_step/period_step/plateau]
        if lrp == "custom_step":
            lr_boundaries = []          # list, [custom_step] Use the specified lr at the given boundaries
        if lrp == "period_step":
            lr_step = 999999999         # int, [period_step] Decay the base learning rate at a fixed step
        if lrp in ["custom_step", "period_step", "plateau"]:
            lr_rate = 0.1               # float, [period_step, plateau] Learning rate decay rate
        if lrp in ["plateau", "cosine", "poly", "cosinev2"]:
            lr_end = 0.                 # float, [plateau, cosine, poly] The minimal end learning rate
        if lrp == "plateau":
            lr_patience = 30            # int, [plateau] Learning rate patience for decay
            lr_min_delta = 1e-4         # float, [plateau] Minimum delta to indicate improvement
            cool_down = 0               # bool, [plateau]
            monitor = "val_loss"        # str, [plateau] Quantity to be monitored [val_loss/loss]
        if lrp == "poly":
            power = 0.9                 # float, [poly]
        if lrp == 'cosinev2':
            lr_repeat = 2
            lr_rev = False
        optim = "sgd"                   # str, Optimizer for training [sgd/adam|sam]
        if optim == "adam":
            adam_beta1 = 0.9            # float, [adam] Parameter
            adam_beta2 = 0.999          # float, [adam] Parameter
            adam_epsilon = 1e-8         # float, [adam] Parameter
        if optim in ["sgd", "sam"]:
            sgd_momentum = 0.9          # float, [momentum] Parameter
            sgd_nesterov = False        # bool, [momentum] Parameter
        weight_decay = 0.00005          # float, weight decay coefficient

        # Loss configurations ================================================================
        loss = "cedt"                   # str, loss type. ce: xentropy; cedt: xentropy with a weight map. [ce/cedt]
        sigma = 5.                      # float, sigma value used in DT loss
        precompute_weight = True        # bool, precompute weights for accelerating training (2x)
        pair_lossW = 0.02               # float, loss weight for pairwise loss

        # Network configurations ====================================================================
        network = "fptrans"             # str, network name
        backbone = "ViT-B/16-384"       # str, structure of the feature extractor.
        drop_rate = 0.1                 # float, drop rate used in the DropBlock of the purifier
        block_size = 4                  # int, block size used in the DropBlock of the purifier
        drop_dim = 1                    # int, 1 for 1D Dropout, 2 for 2D DropBlock
        print_model = False             # bool, print model structure before experiments

        # ViT/DeiT
        vit_stride = None
        vit_depth = 10

        # Prompts
        bg_num = 5                      # int, number of background proxies
        num_prompt = 12 * (1 + bg_num * shot)   # int, number of prompts
        pt_std = 0.02                   # float, standard deviation of initial prompt tokens (Gaussian)

        # Structure of a single episode used in `predict` command
        p = {
            "cls": -1,                  # int, image class, specify the used index in the support mask
            "sup": "",                  # str, support image stem, only for predefined datasets
            "qry": "",                  # str, query image stem, only for predefined datasets
            "sup_rgb": "",              # str, support image path, for custom dataset
            "sup_msk": "",              # str, support mask path, for custom dataset
            "qry_rgb": "",              # str, query image path, for custom dataset
            "qry_msk": "",              # str, [optional] query mask path, for custom dataset
            "out": "",                  # str, path to saving the model prediction
            "overlap": True,            # bool, overlap the predicted mask on the query image
        }
        save_dir = None

    ex.add_source_file("networks/__init__.py")

    @ex.config_hook
    def config_hook(config, command_name, logger):
        if command_name in ["train", "test"]:
            if config["split"] == -1:
                raise ValueError("Argument `split` is required! For example: `split=0` ")

            add_observers(ex, config, fileStorage=config["fileStorage"], MongoDB=config["mongodb"], db_name=ex.path)
            ex.logger = get_global_logger(name=ex.path)
        return config

    return ex


def add_observers(ex, config, fileStorage=True, MongoDB=True, db_name="default"):
    if fileStorage:
        observer_file = FileStorageObserver(config["log_dir"])
        ex.observers.append(observer_file)

    if MongoDB:
        try:
            host, port = config["mongo_host"], config["mongo_port"]
            observer_mongo = MongoObserver(url=f"{host}:{port}", db_name=db_name)
            ex.observers.append(observer_mongo)
        except ModuleNotFoundError:
            # Ignore Mongo Observer
            pass


def init_environment(ex, _run, _config, *args, eval_after_train=False):
    configs = [_config] + list(args)
    for i in range(len(configs)):
        configs[i] = MapConfig(configs[i])
    opt = configs[0]
    logger = get_global_logger(name=ex.path)
    ex.logger = logger

    if not eval_after_train:
        # Create experiment directory
        run_dir = Path(output_dir) / str(_run._id)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'RUN DIRECTORY: {run_dir}')
        _run.run_dir = run_dir

        # Backup source code
        recover_backup_names(_run)

        # Reproducbility
        set_seed(opt.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
        # cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    logger.info('Run:' + ' '.join(sys.argv))
    logger.info(f"Init ==> split {opt.split}, shot {opt.shot}")
    return *configs, logger, device


def recover_backup_names(_run):
    if _run.observers:
        for obs in _run.observers:
            if isinstance(obs, FileStorageObserver):
                for source_file, _ in _run.experiment_info['sources']:
                    Path(f'{obs.dir}/source/{source_file}').parent.mkdir(parents=True, exist_ok=True)
                    obs.save_file(source_file, f'source/{source_file}')
                shutil.rmtree(f'{obs.basedir}/_sources')

                # Convert directory `source` to a tarfile `source.tar.gz` for saving server nodes
                with tarfile.open(f"{obs.dir}/source.tar.gz", "w:gz") as t:
                    for root, dir, files in os.walk(f"{obs.dir}/source"):
                        # print(root, dir, files)
                        for f in files:
                            fullpath = os.path.join(root, f)
                            t.add(fullpath, arcname='/'.join(fullpath.split('/')[2:]))
                shutil.rmtree(f'{obs.dir}/source')
                break


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MapConfig(ReadOnlyDict):
    """
    A wrapper for dict. This wrapper allow users to access dict value by `dot` operation.
    For example, you can access `opt["split"]` by `opt.split`, which makes the code more clear.

    Notice that the result object is a sacred.config.custom_containers.ReadOnlyDict, which is
    a read-only dict for preserving the configuration.

    Parameters
    ----------
    obj: ReadOnlyDict
        Configuration dict.
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = MapConfig(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(MapConfig, self).__init__(new_dict, **kwargs)
