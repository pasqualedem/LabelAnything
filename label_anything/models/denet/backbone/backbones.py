from collections import OrderedDict
import torch.nn as nn

from torchvision.models.vgg import vgg16, vgg16_bn
from torchvision.models.resnet import resnet50, resnet101
from torchvision.models.densenet import densenet121


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class VGG16Base(nn.Module):
    def __init__(self,
                 pretrained=True,
                 replace_stride_with_dilation=None,
                 use_as_based=True,
                 ceil_mode=False):
        super(VGG16Base, self).__init__()
        self.use_as_based = use_as_based
        model = vgg16(pretrained=pretrained)
        features = model.features
        self.conv1 = features[0:4]
        self.pool1 = features[4]
        self.pool1.ceil_mode = ceil_mode
        self.conv2 = features[5:9]
        self.pool2 = features[9]
        self.pool2.ceil_mode = ceil_mode

        if use_as_based:
            self.conv3 = features[10:16]
            self.pool3 = features[16]
            self.pool3.ceil_mode = ceil_mode
            self.conv4 = features[17:23]
            self.pool4 = features[23]
            self.pool4.ceil_mode = ceil_mode
            self.conv5 = features[24:30]
            self.pool5 = features[30]
            self.pool5.ceil_mode = ceil_mode
        else:
            self.layer1 = features[10:17]
            self.layer2 = features[17:24]
            self.layer3 = features[24:]

            self.dilation = 1
            self.replace_stride_with_dilation = replace_stride_with_dilation
            if self.replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                self.replace_stride_with_dilation = [False, False, False]
            if len(self.replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(self.replace_stride_with_dilation))

            self._dilate()

    def _dilate(self):
        def _set_dilate(layer, dilation):
            cnt = -1
            if dilation:
                self.dilation *= 2
                for m in layer.__dict__['_modules'].values():
                    cnt += 1
                    if cnt == 2:
                        m.dilation = (self.dilation, self.dilation)
                        m.padding = (self.dilation, self.dilation)
                return layer[:-1]
            else:
                return layer

        self.layer1 = _set_dilate(self.layer1, self.replace_stride_with_dilation[0])
        self.layer2 = _set_dilate(self.layer2, self.replace_stride_with_dilation[1])
        self.layer3 = _set_dilate(self.layer3, self.replace_stride_with_dilation[2])

    def forward(self, x):
        if not self.use_as_based:
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            return x
        else:
            raise ValueError('forward func is not available when use as based.')


class VGG16BNBase(nn.Module):
    def __init__(self, ceil_mode=False):
        super(VGG16BNBase, self).__init__()
        model = vgg16_bn(pretrained=True)
        features = model.features
        self.conv1 = features[0:6]
        self.pool1 = features[6]
        self.pool1.ceil_mode = ceil_mode

        self.conv2 = features[7:13]
        self.pool2 = features[13]
        self.pool2.ceil_mode = ceil_mode

        self.conv3 = features[14:23]
        self.pool3 = features[23]
        self.pool3.ceil_mode = ceil_mode

        self.conv4 = features[24:33]
        self.pool4 = features[33]
        self.pool4.ceil_mode = ceil_mode

        self.conv5 = features[34:43]
        self.pool5 = features[43]
        self.pool5.ceil_mode = ceil_mode

    def forward(self, *input):
        pass


class ResNet50(IntermediateLayerGetter):
    def __init__(self, return_layers=None):
        model = resnet50(
            pretrained=True,
            replace_stride_with_dilation=[False, True, True])
        return_layers = {
                            'layer1': 'layer1',
                            'layer2': 'layer2',
                            'layer3': 'layer3',
                            'layer4': 'layer4',
                        } or return_layers
        super(ResNet50, self).__init__(model, return_layers)


class ResNet101(IntermediateLayerGetter):
    def __init__(self, return_layers=None):
        model = resnet101(
            pretrained=True,
            replace_stride_with_dilation=[False, True, True])
        return_layers = {
                            'layer1': 'layer1',
                            'layer2': 'layer2',
                            'layer3': 'layer3',
                            'layer4': 'layer4',
                        } or return_layers
        super(ResNet101, self).__init__(model, return_layers)


class DenseNet121(IntermediateLayerGetter):
    def __init__(self, return_layers=None):
        model = densenet121(pretrained=True)
        return_layers = {
                            'denseblock1': 'layer1',
                            'denseblock2': 'layer2',
                            'denseblock3': 'layer3',
                        } or return_layers
        # pop the avg pool layer to make sure layer2 and layer3 share has the same spatial size
        model.features.transition2._modules.pop('pool')
        super(DenseNet121, self).__init__(model.features, return_layers)


if __name__ == '__main__':
    def run_vgg():
        import torch
        model = VGG16Base(pretrained=True,
                          replace_stride_with_dilation=[False, True, True],
                          use_as_based=False)
        print(model)
        x = torch.rand(1, 3, 321, 321)
        x = model(x)
        print(x.size())


    def run_resNet50():
        import torch
        from denet.common import freeze_bn, name_size_grad
        x = torch.rand(1, 3, 321, 321)
        test = ResNet50()
        print(test)
        freeze_bn(test)
        name_size_grad(test)
        x = test(x)
        for k, v in x.items():
            print(k, v.size())


    def run_denseNet121():
        import torch

        x = torch.rand(1, 3, 321, 321)
        test = DenseNet121()
        x = test(x)
        for k, v in x.items():
            print(k, v.size())


    def run_resNet101():
        import torch
        from denet.common import freeze_bn, name_size_grad
        x = torch.rand(1, 3, 321, 321)
        test = ResNet101()
        x = test(x)
        for k, v in x.items():
            print(k, v.size())


    # run_vgg()
    run_resNet50()
    # run_resNet101()
    # run_denseNet121()
