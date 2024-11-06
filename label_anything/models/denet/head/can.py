import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import get_backbone
from ..common import freeze_weights, RegASPP
from ..utils import depart_first_dim, merge_first_k_dim


class CAN(nn.Module):
    """
    Class-Agnostic Segmentation Networks with Iterative Refinement and
    Attentive Few-Shot Learning (CAN).
    http://arxiv.org/abs/1903.02351
    """

    def __init__(self,
                 backbone='ResNet50',
                 refine_time=3):
        super(CAN, self).__init__()
        assert backbone in ['ResNet50',
                            'DenseNet121'], 'get un supported backbone `%s` for CAN.' % backbone
        self.relu = nn.ReLU(inplace=False)
        self.backbone = get_backbone(backbone)()
        assert refine_time >= 1, 'refine time must >= 1, but got `%d`' % refine_time
        self.refine_time = refine_time
        freeze_weights(self.backbone)

        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=0.5))

        self.fuse_history = nn.Sequential(
            nn.Conv2d(256 + 2, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1))

        self.residual1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.aspp = RegASPP()

        self.classifier = nn.Conv2d(256, 2, kernel_size=1)
        self.feature_reuse = None

    def forward(self, *input):
        Is, Ys, Iq = input
        N, way, shot = Is.size()[:3]
        assert way == 1, "The network only supports 1-way training."

        Out_q = self.backbone(Iq)
        Out_q = self.embedding(torch.cat([Out_q['layer2'], Out_q['layer3']], dim=1))
        h_q, w_q = Out_q.size()[-2:]

        Is = merge_first_k_dim(Is, dims=(0, 1, 2))
        Ys = merge_first_k_dim(Ys, dims=(0, 1, 2))
        Out_s = self.backbone(Is)
        Out_s = self.embedding(torch.cat([Out_s['layer2'], Out_s['layer3']], dim=1))
        h_s, w_s = Out_s.size()[-2:]

        Ys = Ys.unsqueeze(1).float()
        Ys = F.interpolate(Ys, (h_s, w_s), mode='bilinear', align_corners=True)

        fg_position_sum = F.adaptive_avg_pool2d(Ys, (1, 1)) * h_s * w_s + 0.0005
        mask = Ys * Out_s

        mask_avg_pooled = F.adaptive_avg_pool2d(mask, (1, 1)) * h_s * w_s / fg_position_sum
        mask_avg_pooled = mask_avg_pooled.expand(-1, -1, h_q, w_q)

        # N, shot, c, h, w
        mask_avg_pooled = depart_first_dim(mask_avg_pooled, dims=(N, shot))
        mask_avg_pooleds = [mask_avg_pooled[:, i, :, :, :] for i in range(shot)]

        outs = None
        for mask_avg_pooled in mask_avg_pooleds:

            history_mask = torch.zeros_like(torch.cat([Ys[:N], Ys[:N]], dim=1))

            fuse_identity = self.fuse(torch.cat([Out_q, mask_avg_pooled], dim=1))
            out = None
            for _ in range(self.refine_time):
                out_plus_history = torch.cat([fuse_identity, history_mask], dim=1)
                out = self.fuse_history(out_plus_history) + fuse_identity
                out = self.relu(out)
                out = self.residual(self.residual1, out)
                out = self.residual(self.residual2, out)

                # ASPP
                out = self.aspp(out)
                out = self.classifier(out)
                history_mask = out
            if outs is None:
                outs = out
            else:
                outs += out
        outs = outs / shot
        self.feature_reuse = outs

        return outs

    def residual(self, layer, x):
        identity = x
        out = layer(x)
        return self.relu(identity + out)


if __name__ == '__main__':
    def run_network():
        test = CAN().cuda()
        N, shot, c, h, w = 4, 1, 3, 321, 321
        Is = torch.randn([N, 1, shot, c, h, w]).cuda()
        Ys = torch.randint(0, 2, size=[N, 1, shot, h, w]).cuda()
        Iq = torch.randn([N, c, h, w]).cuda()

        pred = test(Is, Ys, Iq)
        print(pred.size())

    run_network()
