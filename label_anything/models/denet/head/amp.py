import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import get_backbone
from ..common import freeze_weights, NormMaskedAveragePooling
from ..utils import merge_first_k_dim, depart_first_dim


class AMP(nn.Module):
    """
    Adaptive Masked Proxies for Few-Shot Segmentation (AMP)
    http://arxiv.org/abs/1902.11123
    """

    def __init__(self, backbone='VGG16Base', maximum_num_classes=21, depth=256):
        super(AMP, self).__init__()
        assert backbone in [None, 'VGG16BNBase',
                            'VGG16Base'], 'get un supported backbone `%s` for AMP.' % backbone
        for k, v in get_backbone(backbone)(ceil_mode=True).__dict__.items():
            if k == '_modules':
                self.__setattr__(k, v)

        self.num_classes = maximum_num_classes
        self.depth = depth
        self.nmap = NormMaskedAveragePooling(depth=self.depth, project=False)

        # set conv4 modules's three conv2d to padding and dilation to (1, 1), (1, 1), (2, 2)
        conv4_modules = []
        conv4_count = 0
        conv4_padding_dilation = [(1, 1), (1, 1), (2, 2)]
        for m in self.conv4:
            if type(m) == nn.Conv2d:
                padding, dilation = conv4_padding_dilation[conv4_count]
                conv4_count += 1
                new_m = nn.Conv2d(in_channels=m.in_channels, out_channels=m.out_channels,
                                  kernel_size=m.kernel_size, stride=m.stride,
                                  padding=padding, dilation=dilation, groups=m.groups,
                                  bias=m.bias is not None)
                new_m.weight.data = m.weight.data
                if m.bias is not None:
                    new_m.bias.data = m.bias.data
                m = new_m
            conv4_modules.append(m)
        self.conv4 = nn.Sequential(*conv4_modules)

        # set conv5 modules's three conv2d to padding and dilation to (2, 2), (4, 4), (4, 4)
        conv5_modules = []
        conv5_count = 0
        conv5_padding_dilation = [(2, 2), (4, 4), (4, 4)]
        for m in self.conv5:
            if type(m) == nn.Conv2d:
                padding, dilation = conv5_padding_dilation[conv5_count]
                conv5_count += 1
                new_m = nn.Conv2d(in_channels=m.in_channels, out_channels=m.out_channels,
                                  kernel_size=m.kernel_size, stride=m.stride,
                                  padding=padding, dilation=dilation, groups=m.groups,
                                  bias=m.bias is not None)
                new_m.weight.data = m.weight.data
                if m.bias is not None:
                    new_m.bias.data = m.bias.data
                m = new_m
            conv5_modules.append(m)
        self.conv5 = nn.Sequential(*conv5_modules)

        freeze_weights(nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5))

        self.shot = 1

        # DFCN8s setting
        # self.fully_conv = nn.Sequential(
        #     nn.Conv2d(self.depth * 2, 4096, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(),
        #     nn.Conv2d(4096, 4096, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(),
        #     nn.Conv2d(4096, self.depth, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d()
        # )

        # reduced DFCN8s setting
        self.fully_conv = nn.Sequential(
            nn.Conv2d(2 * self.depth, self.depth, 1)
        )

        self.level1 = nn.Sequential(self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.pool3)
        self.level2 = nn.Sequential(self.conv4, self.pool4)
        self.level3 = nn.Sequential(self.conv5, self.pool5, self.fully_conv)

        self.level1_classifier = nn.Conv2d(self.depth, self.num_classes, 1, bias=False)
        self.level2_classifier = nn.Conv2d(self.depth * 2, self.num_classes, 1, bias=False)
        self.level3_classifier = nn.Conv2d(self.depth, self.num_classes, 1, bias=False)

    def forward(self, *x, **kwargs):
        if self.training:
            return self._train_forward(*x)
        else:
            return self._val_forward(*x, **kwargs)

    def _train_forward(self, x):
        level1 = self.level1(x)
        pre_level1 = self.level1_classifier(level1)
        level2 = self.level2(level1)
        pre_level2 = self.level2_classifier(level2)
        level3 = self.level3(level2)
        level3 = self.level3_classifier(level3)

        out = pre_level2 + F.interpolate(level3, size=level2.size()[-2:], mode='bilinear', align_corners=True)
        out = pre_level1 + F.interpolate(out, size=level1.size()[-2:], mode='bilinear', align_corners=True)
        return out

    def _val_forward(self, *x, **kwargs):
        Is, Ys, Iq, label = x
        N, way, shot = Is.size()[:3]
        assert way == 1, "The network only supports 1-way training."
        self.shot = shot
        Is = merge_first_k_dim(Is, dims=(0, 1, 2))
        Ys = merge_first_k_dim(Ys, dims=(0, 1, 2))
        alpha = kwargs.get('alpha', 0.25821)
        base = kwargs.get('base', True)

        Fs_level1 = self.level1(Is)
        weight = self.level1_classifier.weight[0].clone().detach()
        update_weight, return_weight = self.update_proxies(Fs_level1, Ys, alpha, weight)
        self.level1_classifier.weight[0] = update_weight.reshape([self.depth, 1, 1])
        Fq_level1 = self.level1(Iq)
        if base:
            logits1 = self.full_logits(
                self.level1_classifier.weight.clone().detach(),
                label, Fq_level1, Fs_level1, Ys
            )
        else:
            logits1 = self.classifier(Fq_level1, Fs_level1, Ys, return_weight)

        Fs_level2 = self.level2(Fs_level1)
        weight = self.level2_classifier.weight[0].clone().detach()
        update_weight, return_weight = self.update_proxies(Fs_level2, Ys, alpha, weight)
        self.level2_classifier.weight[0] = update_weight.reshape([self.depth * 2, 1, 1])
        Fq_level2 = self.level2(Fq_level1)
        if base:
            logits2 = self.full_logits(
                self.level2_classifier.weight.clone().detach(),
                label, Fq_level2, Fs_level2, Ys
            )
        else:
            logits2 = self.classifier(Fq_level2, Fs_level2, Ys, return_weight)

        Fs_level3 = self.level3(Fs_level2)
        weight = self.level3_classifier.weight[0].clone().detach()
        update_weight, return_weight = self.update_proxies(Fs_level3, Ys, alpha, weight)
        self.level3_classifier.weight[0] = update_weight.reshape([self.depth, 1, 1])
        Fq_level3 = self.level3(Fq_level2)
        if base:
            logits3 = self.full_logits(
                self.level3_classifier.weight.clone().detach(),
                label, Fq_level3, Fs_level3, Ys
            )
        else:
            logits3 = self.classifier(Fq_level3, Fs_level3, Ys, return_weight)

        out = logits2 + F.interpolate(logits3, size=logits2.size()[-2:], mode='bilinear', align_corners=True)
        out = logits1 + F.interpolate(out, size=logits1.size()[-2:], mode='bilinear', align_corners=True)
        return out

    def update_proxies(self, embedding, mask, alpha, previous_weight):
        N, c = embedding.size()[:2]
        proxies = self.nmap(embedding, 1 - mask).reshape([-1, c])  # background proxy
        update_weight = previous_weight.reshape([-1, c])  # previous background weights
        return_weight = []
        for proxy, cnt in zip(torch.chunk(proxies, N, 0), range(N)):
            update_weight = alpha * proxy + (1 - alpha) * update_weight
            if (cnt + 1) % self.shot == 0:
                return_weight.append(update_weight)
        return_weight = torch.cat(return_weight, dim=0).unsqueeze(-1)
        return update_weight.transpose(0, 1), return_weight

    def classifier(self, Fq, Fs, Ys, weight):
        N, _, h, w = Fq.size()
        fg_weight = self.nmap(Fs, Ys).reshape([N * self.shot, -1, 1])
        fg_weight = depart_first_dim(fg_weight, dims=(N, self.shot))
        fg_weight = fg_weight.mean(dim=1, keepdim=False)
        linear = torch.cat([weight, fg_weight], dim=-1)
        Fq = Fq.permute([0, 2, 3, 1]).reshape([N, h * w, -1])
        return torch.bmm(Fq, linear).transpose(1, 2).reshape([N, -1, h, w])

    def full_logits(self, full_weights, label, Fq, Fs, Ys):
        N, c, h, w = Fq.shape
        full_weights = full_weights.squeeze().transpose(0, 1)
        full_weights = torch.stack([full_weights for _ in range(N)], dim=0)
        novel_weights = self.nmap(Fs, Ys).reshape([N * self.shot, -1])
        novel_weights = depart_first_dim(novel_weights, dims=(N, self.shot))
        novel_weights = novel_weights.mean(dim=1, keepdim=False).detach()
        # print("base mean: ", torch.mean(torch.abs(full_weights)))
        # print("novel mean: ", torch.mean(torch.abs(novel_weights)))
        for _ in range(N):
            full_weights[_, :, label[_]] = novel_weights[_]
        Fq = Fq.reshape([N, c, -1]).permute([0, 2, 1])
        return torch.bmm(Fq, full_weights).transpose(1, 2).reshape([N, -1, h, w])


if __name__ == '__main__':
    def run_network_cpu():
        shot = 1
        model = AMP(backbone='VGG16Base')
        I = torch.randn([4, 3, 321, 321])
        logits = model(I)
        print("train logits: ", logits.shape)
        Is = torch.randn([4, 1, shot, 3, 321, 321])
        Ys = torch.randint(0, 2, size=[4, 1, shot, 321, 321])
        Iq = torch.randn([4, 3, 321, 321])
        logits = model(Is, Ys, Iq)
        model.eval()
        print("val logits: ", logits.shape)


    def run_network_gpu():
        shot = 5
        model = AMP(backbone='VGG16Base')
        model = model.cuda()
        I = torch.randn([4, 3, 321, 321]).cuda()
        logits = model(I)
        print("train logits: ", logits.shape)
        Is = torch.randn([4, 1, shot, 3, 321, 321]).cuda()
        Ys = torch.randint(0, 2, size=[4, 1, shot, 321, 321]).cuda()
        Iq = torch.randn([4, 3, 321, 321]).cuda()
        cls = torch.randint(6, 10, size=[4])
        model.eval()
        logits = model(Is, Ys, Iq, cls)
        print("val logits: ", logits.shape)


    # run_network_cpu()
    run_network_gpu()
