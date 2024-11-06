import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import get_backbone
from ..common import freeze_weights, RegASPP, GAU
from ..utils import merge_first_k_dim


class PGN(nn.Module):
    """
    Pyramid Graph Networks with Connection Attentions for Region-Based
    One-Shot Semantic Segmentation (PGN).
    ICCV, 2019.
    """

    def __init__(self, backbone="ResNet50"):
        super(PGN, self).__init__()
        assert backbone in ['ResNet50',
                            'DenseNet121'], 'get unsupported backbone "%s" for GPN.' % backbone
        DEPTH = 256
        self.backbone = get_backbone(backbone)()
        freeze_weights(self.backbone)

        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

        self.g = nn.Sequential(nn.Conv2d(in_channels=DEPTH, out_channels=DEPTH, kernel_size=1),
                               nn.ReLU(inplace=True))

        self.gau_1 = GAU(depth=DEPTH)
        self.gau_2 = GAU(depth=DEPTH)
        self.gau_3 = GAU(depth=DEPTH)
        self.gau_4 = GAU(depth=DEPTH)

        self.residual1 = nn.Sequential(
            nn.Conv2d(in_channels=DEPTH, out_channels=DEPTH, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=DEPTH, out_channels=DEPTH, kernel_size=3, padding=1)
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(in_channels=DEPTH, out_channels=DEPTH, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=DEPTH, out_channels=DEPTH, kernel_size=3, padding=1)
        )

        self.residual3 = nn.Sequential(
            nn.Conv2d(in_channels=DEPTH, out_channels=DEPTH, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=DEPTH, out_channels=DEPTH, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU(inplace=False)

        self.aspp = RegASPP()
        self.classifier = nn.Conv2d(in_channels=DEPTH, out_channels=2, kernel_size=1)
        self.feature_reuse = None

    def forward(self, *input):
        Is, Ys, Iq = input
        assert Is.size()[1] == 1, "The network only supports 1-way training."

        Is = merge_first_k_dim(Is, dims=(0, 1, 2))
        Ys = merge_first_k_dim(Ys, dims=(0, 1, 2))
        if len(list(Ys.size())) == 3:
            Ys = Ys.unsqueeze(1)
        Ys = Ys.float()
        Out_q = self.backbone(Iq)
        Fq = self.embedding(torch.cat([Out_q['layer2'], Out_q['layer3']], dim=1))
        h_q, w_q = Fq.size()[-2:]

        Out_s = self.backbone(Is)
        Fs = self.embedding(torch.cat([Out_s['layer2'], Out_s['layer3']], dim=1))
        Ys = F.interpolate(Ys, size=Fs.size()[-2:], mode='nearest')

        # insert supervision here, add a very small term
        Fs = Ys * Fs + 0.0005

        Gs, Gq = self.g(Fs), self.g(Fq)

        Fq_d4 = F.adaptive_avg_pool2d(Fq, (h_q // 4, w_q // 4))
        Fq_d2 = F.adaptive_avg_pool2d(Fq, (h_q // 2, w_q // 2))
        Fq_d1 = Fq

        Fq_d4_out = self.gau_1(Fs, Fq_d4, Ys, Gs, Gq)
        Fq_d2_out = self.gau_2(Fs, Fq_d2, Ys, Gs, Gq)
        Fq_d1_out = self.gau_3(Fs, Fq_d1, Ys, Gs, Gq)
        Fq_global_out = self.gau_4(Fs, Fq_d1, Ys, Gs, Gq, equal_weight=True)

        Fq_d4_out = F.interpolate(Fq_d4_out, (h_q, w_q), mode='bilinear', align_corners=False)
        Fq_d2_out = F.interpolate(Fq_d2_out, (h_q, w_q), mode='bilinear', align_corners=False)

        out = Fq_d4_out + Fq_d2_out + Fq_d1_out + Fq_global_out

        out = self.residual(self.residual1, out)
        out = self.residual(self.residual2, out)
        out = self.residual(self.residual3, out)

        out = self.aspp(out)

        self.feature_reuse = out
        out = self.classifier(out)

        return out

    def residual(self, layer, x):
        identify = x
        out = layer(x)
        return self.relu(identify + out)


if __name__ == '__main__':
    def run_network():
        test = PGN().cuda()
        N, shot, c, h, w = 4, 1, 3, 321, 321
        Is = torch.randn([N, 1, shot, c, h, w]).cuda()
        Ys = torch.randint(0, 2, size=[N, 1, shot, h, w]).cuda()
        Iq = torch.randn([N, c, h, w]).cuda()

        pred = test(Is, Ys, Iq)
        print(pred.size())


    run_network()
