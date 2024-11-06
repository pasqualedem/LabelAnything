from einops import rearrange
from label_anything.data.utils import BatchKeys
from .backbone import *
from .head import *
from .common import *

from . import DENet as OriginalDENet


class DeNet(OriginalDENet):
    def forward(self, batch: dict):
        images = batch[BatchKeys.IMAGES]
        Iq = images[:, 0]
        Is = images[:, 1:]
        Ys = batch[BatchKeys.PROMPT_MASKS] # B M C H W
        Ys = Ys.argmax(dim=2)
        b, m, c, _, _ = Ys.shape
        c_fg = c - 1
        k = m // c_fg
        # Is: (B, way, shot, 3, H, W)
        Is = rearrange(Is, 'b (k c) rgb h w -> b c k rgb h w', k=k)
        Ys = rearrange(Ys, 'b (k c) h w -> b k c m h w')
        return super().forward(Is, Ys, Iq, label)