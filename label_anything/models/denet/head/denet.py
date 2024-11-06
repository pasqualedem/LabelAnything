import torch
import torch.nn as nn
from ..backbone import get_backbone
from ..common import DeepLabHead, freeze_weights, WeightEstimator
from ..utils import merge_first_k_dim, depart_first_dim, get_binary_logits


class DENet(nn.Module):
    """
    Dynamic Extension Nets (DENet).

    Args:
        backbone: use which backbone to extract features. For now, `ResNet50`,
            `DenseNet121` are available.
        maximum_num_classes: specify maximum number of classes, default 21 for
            VOC dataset, when using COCO dataset, set it to 81.
        depth: interval channel depth for every feature map, after feature
            extracted from backbone.
        drop_rate: dropout rate.
        support_free: whether to use `support free` mode.
        visualize: whether to visualize the hidden feature maps.
    """

    def __init__(self,
                 backbone="ResNet50",
                 maximum_num_classes=21,
                 depth=256,
                 drop_rate=0.5,
                 support_free=False,
                 visualize=True):
        super(DENet, self).__init__()
        assert backbone in ['ResNet50', 'DenseNet121'], \
            'get unsupported backbone "%s" for DENet.' % backbone

        self.maximum_num_classes = maximum_num_classes
        self.depth = depth
        self.visualize = visualize
        self.vis = dict()

        self.backbone = get_backbone(backbone)()
        freeze_weights(self.backbone)

        self.embedding = nn.Sequential(
            nn.Conv2d(1024, self.depth, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate))
        self.deeplab_head = DeepLabHead(256)

        self.estimator = WeightEstimator(self.maximum_num_classes, self.depth,
                                         support_free=support_free)

    def forward(self, *x):
        Is, Ys, Iq, label = x
        N, way, shot = Is.size()[:3]
        # Is: (N, way, shot, 3, h, w) -> (N * way * shot, 3, h, w)
        Is = merge_first_k_dim(Is, dims=(0, 1, 2))
        # guarantee that the size of `label` is (N, way)
        if way == 1:
            label = label.unsqueeze(-1)
        out_s = self.backbone(Is)['layer3']
        Fs = self.embedding(out_s)
        Fs = self.deeplab_head(Fs, relu=True)
        # Fs: (N * way * shot, c, h, w) -> (N * way, shot, 3, h, w)
        Fs = depart_first_dim(Fs, dims=(N * way, shot))
        if self.visualize and way == 1:
            self.vis.update({'hidden_Fs': torch.select(Fs, dim=1, index=0).clone().detach()})
        # Fs: (N * way, shot, 3, h, w) -> (N, way, shot, 3, h, w)
        Fs = depart_first_dim(Fs, dims=(N, way))

        # forward query
        out_q = self.backbone(Iq)['layer3']
        Fq = self.embedding(out_q)
        Fq = self.deeplab_head(Fq, relu=True)

        if self.visualize:
            self.vis.update({'hidden_Fq': Fq.clone().detach()})

        # get knowledge logits
        logits_full = self.estimator(Fq, Fs, Ys, label, mode='training')

        # get pattern logits
        if way > 1:
            logits_binary = []
            for i in range(way):
                l = torch.select(label, dim=1, index=i)
                logits_binary.append(get_binary_logits(logits_full, l))
            logits_binary = torch.stack(logits_binary, dim=1)
            logits_binary = merge_first_k_dim(logits_binary, (0, 1))
        else:
            label = torch.select(label, dim=1, index=0)
            logits_binary = get_binary_logits(logits_full, label, base=True)

        return logits_full, logits_binary


if __name__ == "__main__":
    def run_network_cpu():
        N, way, shot, c, h, w = 4, 1, 5, 3, 224, 224
        Is = torch.randn([N, way, shot, c, h, w])
        Iq = torch.randn([N, c, h, w])
        Ys = torch.randint(0, 2, size=[N, way, shot, h, w])
        cls = torch.randint(6, 10, size=[N])
        if way > 1:
            cls = torch.randint(6, 10, size=[N, way]).cuda()
        print("cls: ", cls)
        model = DENet()
        model = model
        k, p = model(Is, Ys, Iq, cls)
        print(k.shape)
        print(p.shape)


    def run_network_gpu():
        N, way, shot, c, h, w = 2, 1, 1, 3, 224, 224
        Is = torch.randn([N, way, shot, c, h, w]).cuda()
        Iq = torch.randn([N, c, h, w]).cuda()
        Ys = torch.randint(0, 2, size=[N, way, shot, h, w]).cuda()
        cls = torch.randint(6, 10, size=[N]).cuda()
        if way > 1:
            cls = torch.randint(6, 10, size=[N, way]).cuda()
        print("cls: ", cls)
        model = DENet()
        model = model.cuda()
        k, p = model(Is, Ys, Iq, cls)
        print(k.shape)
        print(p.shape)


    # run_network_cpu()
    run_network_gpu()
