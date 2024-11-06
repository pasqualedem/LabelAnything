from .can import CAN
from .pgn import PGN
from .amp import AMP
from .denet import DENet

key2arch = {
    'CAN': CAN,
    'PGN': PGN,
    'AMP': AMP,
    'DENet': DENet,
}


def get_architecture(arch_name=None):
    if arch_name is None:
        print("Using default DENet model")
        return DENet

    else:
        if arch_name not in key2arch:
            raise NotImplementedError('Model {} not implemented'.format(arch_name))

        print('Using model: {}'.format(arch_name))
        return key2arch[arch_name]
