from .backbones import VGG16Base, VGG16BNBase
from .backbones import ResNet50, ResNet101
from .backbones import DenseNet121

__all__ = [
    'VGG16BNBase',
    'VGG16Base',
    'ResNet50',
    'ResNet101',
    'DenseNet121',
    'get_backbone',
]

key2backbone = {
    'VGG16BNBase': VGG16BNBase,
    'VGG16Base': VGG16Base,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'DenseNet121': DenseNet121,
}


def get_backbone(backbone_name):
    if backbone_name is None:
        print("Using default VGG16Base backbone")
        return VGG16Base

    else:
        if backbone_name not in key2backbone:
            raise NotImplementedError('Backbone {} not implemented'.format(backbone_name))
        print('Using backbone: {}'.format(backbone_name))
        return key2backbone[backbone_name]
