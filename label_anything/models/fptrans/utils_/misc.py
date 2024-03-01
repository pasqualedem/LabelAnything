from functools import partial
from PIL import Image
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F

from label_anything.models.fptrans.constants import on_cloud


interpb = partial(F.interpolate, mode='bilinear', align_corners=True)
interpn = partial(F.interpolate, mode='nearest')


def load_image(name, dtype='img', proc='cv2', mode=None):
    if proc == 'pil':
        img = Image.open(name)
    elif proc == 'cv2':
        if dtype == 'img':
            img = cv2.imread(str(name), mode or cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f'Can not open file {name}.')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(str(name), mode or cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f'Can not open file {name}.')
    else:
        raise ValueError
    return img


def load_weights(name):
    npzfile = np.load(name)
    edts = npzfile['x']
    classes = npzfile['c']
    return {
        'x': edts,
        'c': classes.tolist()
    }


def get_pth(exp_dir, ckpt):
    ckpt_name = [ckpt, "bestckpt.pth", "ckpt.pth"]
    for name in ckpt_name:
        if name is None:
            continue
        ckpt_path = exp_dir / name
        if ckpt_path.exists() and ckpt_path.is_file():
            return ckpt_path
    return False


def find_snapshot(output_dir, exp_id=-1, ckpt=None, afs=False):
    """ Find experiment checkpoint """

    if ckpt and Path(ckpt).exists():
        return Path(ckpt)

    if afs:
        output_dir = Path("./afs/proj/FSS-PyTorch-ori/output")

    ckpt_path = get_pth(output_dir / str(exp_id), ckpt)
    if ckpt_path:
        return ckpt_path

    if not on_cloud:
        # Import readline module to improve the experience of input
        # noinspection PyUnresolvedReferences
        import readline

        while True:
            ckpt_path = Path(input(f"Cannot find checkpoints: {ckpt} or exp_id: {exp_id}. \nPlease input:"))
            if ckpt_path.exists():
                return ckpt_path
