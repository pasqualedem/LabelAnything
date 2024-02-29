from label_anything.models.fptrans.FPTrans import FPTrans
from label_anything.data.utils import BatchKeys
from label_anything.models.fptrans.utils_.misc import interpb, interpn
import torch
import logging


__networks = {
    "fptrans": FPTrans,
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_model(opt, logger, *args, **kwargs):
    if opt.network.lower() in __networks:
        model = __networks[opt.network.lower()](opt, logger, *args, **kwargs)
        if opt.print_model:
            print(model)
        return model
    else:
        raise ValueError(
            f"Not supported network: {opt.network}. {list(__networks.keys())}"
        )


def build_fptrans(
    backbone_checkpoint: str = "checkpoints/backbone.pth",
    model_checkpoint: str = "checkpoints/fptrans.pth",
    dataset: str = "COCO",  # can be "COCO" or "PASCAL"
):
    opt = {
        "shot": 1,
        "drop_dim": 1,
        "drop_rate": 0.1,
        "block_size": 4,
        "backbone": "ViT-B/16-384",
        "tqdm": False,
        "height": 480,
        "bg_num": 5,
        "num_prompt": 72,
        "vit_stride": None,
        "dataset": dataset,
        "coco2pascal": False,
        "pt_std": 0.02,
        "vit_depth": 10,
    }
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    opt = dotdict(opt)
    model = FPTrans(opt, logger, backbone_checkpoint)
    model.load_weights(model_checkpoint, logger)

    return model


class FPTransMultiClass(FPTrans):
    def forward(self, x):
        B, M, Ch, H, W = x[BatchKeys.IMAGES].size()
        C = x[BatchKeys.PROMPT_MASKS].size(2)
        S = M - 1

        q = x[BatchKeys.IMAGES][:, 0]
        s_x = x[BatchKeys.IMAGES][:, 1:]
        s_y = x[BatchKeys.PROMPT_MASKS][:, 1:]  # B, M, C, H, W

        logits = []

        for c in range(C):
            logits.append(super().forward(q, s_x, s_y[:, :, c, :, :], None, None))
            


if __name__ == "__main__":
    model = build_fptrans(
        "checkpoints/B_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz",
        "checkpoints/fptrans.pth",
    )
    print(model)
    print("Done")
