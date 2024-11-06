import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import merge_first_k_dim, depart_first_dim


def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_weights(module):
    for param in module.parameters():
        param.requires_grad = True


def freeze_bn(module):
    for c in module.children():
        if isinstance(c, nn.BatchNorm2d):
            freeze_weights(c)
        else:
            freeze_bn(c)


def name_size_grad(module):
    for name, param in module.named_parameters():
        print('name ->', name, ', size -> ', param.size(), ', requires_grad->', param.requires_grad)


class ASPPConv(nn.Sequential):
    """ASPPConv without BN and with dropout."""

    def __init__(self, in_channels, out_channels, dilation, p=0.5):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout2d(p=p),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ASPPPooling without BN and with dropout."""

    def __init__(self, in_channels, out_channels, p=0.5):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout2d(p=p),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class RegASPP(nn.Module):
    """ASPP remove BN and with dropout."""

    def __init__(self, in_channels=256, atrous_rates=(6, 12, 18), p=0.5):
        super(RegASPP, self).__init__()
        out_channels = 256
        modules = list()
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout2d(p=p)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1, p=p))
        modules.append(ASPPConv(in_channels, out_channels, rate2, p=p))
        modules.append(ASPPConv(in_channels, out_channels, rate3, p=p))
        modules.append(ASPPPooling(in_channels, out_channels, p=p))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(p=p))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class GAU(nn.Module):
    """Graph Attention Unit used by PGNet."""

    def __init__(self, depth=256):
        super(GAU, self).__init__()
        self.phi = nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=1)
        self.Phi = nn.Sequential(
            nn.Conv2d(in_channels=2 * depth, out_channels=depth, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, *input, equal_weight=False, eps=1e-8):
        Fs, Fq, Ys, Gs, Gq = input
        nshot, c, hs, ws = Fs.size()
        N, c, hq, wq = Fq.size()
        shot = nshot // N

        Ts, Tq = self.theta(Fs), self.phi(Fq)
        Ts = depart_first_dim(Ts, dims=(N, shot))
        Ts_reshaped = Ts.permute(0, 1, 3, 4, 2).reshape((N, shot * hs * ws, c))
        Tq_reshaped = Tq.permute(0, 2, 3, 1).reshape((N, hq * wq, c))
        att_map = torch.bmm(Tq_reshaped, Ts_reshaped.transpose(2, 1))
        # att_map_mask = att_map.data.masked_fill_((1 - Ys.reshape((N, 1, hs * ws))).bool(), -eps)
        # att = torch.softmax(att_map_mask, dim=2)
        att = torch.softmax(att_map, dim=2)
        Gs = depart_first_dim(Gs, dims=(N, shot))
        Gs_reshaped = Gs.permute(0, 1, 3, 4, 2).reshape((N, shot * hs * ws, c))
        if equal_weight:
            att.data.fill_(1 / (hs * ws))
        v_q = torch.bmm(att, Gs_reshaped).reshape((N, c, hq, wq))

        Gq = F.adaptive_avg_pool2d(Gq, (hq, wq))
        return self.Phi(torch.cat([v_q, Gq], dim=1))


class DeepLabHead(nn.Module):
    """DeepLabHead, ASPP (RegASPP) followed by one CNN-BN-ReLU."""

    def __init__(self, in_channels):
        super(DeepLabHead, self).__init__()
        self.in_channels = in_channels
        self.aspp = RegASPP(in_channels, (6, 12, 18))
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x, relu=True):
        out = self.aspp(x)
        out = self.conv1(out)
        if relu:
            out = self.relu(self.bn(out))
        return out


class MaskedAveragePooling(nn.Module):
    """
    MaskedAveragePooling. Mask w/ or w/o average pooling.
    If project is True, when w/ pooling, we apply a linear projection
    after the masked and pooled embedding.
    Note that when mask and embedding mismatch in spatial dimension,
    we down/up-sample the mask according to embedding's spatial dim.

    Args:
        depth: feature map's channel dimension.
        project: whether to project the the masked and pooled embedding
            by a linear layer.
    """

    def __init__(self, depth=None, project=True):
        super().__init__()
        self.depth = depth
        self.linear = None
        if project and depth is not None:
            self.linear = nn.Conv2d(in_channels=self.depth, out_channels=self.depth, kernel_size=1)

    def forward(self, embedding, mask, pooling=True, eps=1e-3):
        """
        Mask w/ or w/o pooling.
        If project is True, when w/ pooling, we apply a linear projection
        after the masked and pooled embedding.
        Note that when mask and embedding mismatch in spatial dimension,
        we down/up-sample the mask according to embedding's spatial dim.

        Args:
            embedding: feature map with shape `[N, c, h, w]`.
            mask: feature mask with shape `[N, h, w]`.
            pooling: whether to perform pooling after mask the embedding.
            eps: for numerically stable during pooling.
        Returns:
            masked embedding tensor
            when `pooling` is True: return tensor's shape is `[N, c, 1, 1]`,
            else: `[N, c, h, w]`.
        """
        h, w = embedding.size()[-2:]
        mask = mask.unsqueeze(1).float()
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        if pooling:
            numerator = torch.sum(mask * embedding, dim=(2, 3), keepdim=True)
            denominator = torch.sum(mask, dim=(2, 3), keepdim=True)
            prototype = numerator / (denominator + eps)
            if self.linear is not None:
                return self.linear(prototype)
            else:
                return prototype
        else:
            return mask * embedding


class NormMaskedAveragePooling(MaskedAveragePooling):
    """Normalized MAP. Apply l2 normalization after MAP."""

    def forward(self, embedding, mask, pooling=True, eps=1e-3):
        out = super().forward(embedding, mask, pooling, eps)
        return F.normalize(out, p=2, dim=1)


class GAM(nn.Module):
    """
    Guided Attention Module (GAM).

    Args:
        in_channels: interval channel depth for both input and output
            feature map.
        drop_rate: dropout rate.
    """

    def __init__(self, in_channels, drop_rate=0.5):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]
        mask = mask.unsqueeze(1).float()
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        return mask * embedding

    def forward(self, *x):
        Fs, Ys = x
        att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs


class WeightEstimator(nn.Module):
    """Weight estimator used by DENet."""

    def __init__(self,
                 num_classes,
                 depth,
                 drop_rate=0.5,
                 support_free=False):
        """
        Constructor function of weight estimator.

        Args:
            num_classes: the number of classes the weight estimator holds.
            depth: the dimension of the class weight.
            drop_rate: dropout rate.
            support_free: whether to apply 'support free' mode.
        """

        super(WeightEstimator, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.sfree = support_free

        self.gam = GAM(in_channels=depth)
        self.map = MaskedAveragePooling(depth=depth)
        self.dropout = nn.Dropout(p=drop_rate)

        # initialize the class weights
        self.weight = nn.Parameter(0.01 * torch.randn([num_classes, depth]), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # record whether the weight for a class has been inserted
        # when using 'support free' mode
        self.record = {_: False for _ in range(num_classes)}

    def _knowledge_source(self, label, detach=False):
        """
        Get the information from learned base weight source. Select `n-1` prototypes
        from `n` prototypes and hold out the `1` prototype according to label.

        Args:
            label: label used to hold out `1` prototype, with shape `[1]`
            detach: whether to detach the gradient of selected `n-1` prototypes.

        Returns:
            knowledge: The information from knowledge source, with shape of `[n-1, c]`.
        """

        if detach:
            weight = self.weight.clone().detach()
        else:
            weight = self.weight.clone()
        knowledge = weight[torch.arange(weight.size(0)) != label]
        return knowledge

    def _extend(self, label, prototype):
        """
        Get knowledge prototypes while holding out the corresponding class's
        prototype.

        Args:
            label: label used to hold out `1` prototype, with shape `[way]`
            prototype: prototype got from support feature map, with shape
                `[way, shot, c, 1, 1]`.
        Returns:
            updated classifier weight with shape `[maximum_num_classes, hidden_dim]`.
        """

        weight = self.weight.clone()
        # prototype: (way, shot, c, 1, 1) -> (way, c, 1, 1)
        prototype = torch.mean(prototype, dim=1, keepdim=False)
        prototype = prototype.reshape(prototype.size()[:-2])

        # do not update the 'learned classes' prototypes too much is vital
        # if not, the model will find a shortcut that directly learn the base classes
        # as deploy segmentation instead of first maximizing the learning to learn ability
        # then learn the base classes.
        weight = self.dropout(weight)
        for _ in range(len(label)):
            l = label[_].item()
            # only update the weight for the corresponding class when
            # 1) support-based predictions, or
            # 2) support-free predictions and the weight hasn't been inserted
            if not self.sfree or not self.record[l]:
                p = prototype[_]
                weight[l] = p
                # record the corresponding class weight has been inserted
                self.record[l] = True
                if self.sfree:
                    self.weight.data = weight
        return weight

    def extend(self, Fs, Ys, labels, mode="training"):
        """
        Update the corresponding class weights specified by labels given support set.

        Args:
            Fs: feature maps of support images, with shape (N, way, shot, c, h, w).
            Ys: binary mask of support images, with shape (N, way, shot, h, w).
            labels: specify the novel classes to be learned, with shape (N, way).
            mode:
                when `training`, given batch of data, return batch of `cold updated` weights.
                when `deploy`, given batch of data, return sum of batch of `cold updated` weights.
        Returns:
            when `training`, return batch weights, shape (N, num_class, c).
            when `deploy`, return weight, shape (num_class, c).
        """

        assert mode in ['training', 'deploy'], 'mode `%s` not supported' % mode

        if mode == 'training':
            N, way, shot = Fs.size()[:3]
            Fs_merged = merge_first_k_dim(Fs, dims=(0, 1, 2))
            Ys_merged = merge_first_k_dim(Ys, dims=(0, 1, 2))
            Fs_merged = self.gam(Fs_merged, Ys_merged)
            prototypes = self.map(Fs_merged, Ys_merged)
            # prototypes: (N*way*shot, c, h, w) -> (N, way, shot, c, 1, 1)
            prototypes = depart_first_dim(prototypes, dims=(N * way, shot))
            prototypes = depart_first_dim(prototypes, dims=(N, way))
            return torch.stack([self._extend(l, p) for l, p in zip(labels, prototypes)], dim=0)
        if mode == 'deploy':
            batch_weights = self.extend(Fs, Ys, labels, mode='training')
            weight = torch.mean(batch_weights, dim=0, keepdim=False)
            return weight

    def infer(self, embedding, weight, mode="training"):
        """
        Inference operation. After obtaining the novel class weight, we can use this
        function to do inference.

        Args:
            embedding: feature maps of support images, with shape (N, c, h, w).
            weight: weight of shape (N, num_class, c) or (num_class, c).
            mode:
                when `training`, provide batch of weights with shape (N, num_class, c).
                when `deploy`, provide weight with shape (num_class, c).
        Returns:
              logits with shape (N, c, h, w).
        """

        assert mode in ['training', 'deploy'], 'mode `%s` not supported' % mode
        if mode == 'training':
            N, _, h, w = embedding.size()
            embedding = embedding.permute([0, 2, 3, 1]).reshape([N, h * w, -1])
            logits = torch.bmm(embedding, weight.transpose(1, 2)).transpose(1, 2).reshape([N, -1, h, w])
            return logits
        if mode == 'deploy':
            N, _, h, w = embedding.size()
            embedding = embedding.permute([0, 2, 3, 1]).reshape([N, h * w, -1])
            logits = torch.matmul(embedding, self.weight.transpose(0, 1)).transpose(1, 2).reshape([N, -1, h, w])
            return logits

    def forward(self, *x, mode='training'):
        """
        Forward function of weight estimator.

        Args:
            *x:
                embedding, Fs, Ys, labels when mode is `training`.
                embedding, Fs, Ys, labels or embedding when mode is `deploy`.

                embedding: the feature map to be segmented, with shape (N, c, h, w).
                Fs: feature maps of support images, shape (N, way, shot, c, h, w).
                Ys: binary mask of support images, shape (N, way, shot, h, w).
                labels: labels of query classes, shape (N, way).
              mode: `training` or `deploy`,
                when using `training`, the updated weight will not be kept when evaluation.
                when using `deploy`, the updated weight will be remembered.
        Returns:
            logits with shape (N, c, h, w).
        """
        assert mode in ['training', 'deploy'], 'mode `%s` not supported' % mode
        if mode == 'training':
            embedding, Fs, Ys, labels = x
            # (N, num_class, c)
            weights = self.extend(Fs, Ys, labels, mode=mode)
            logits = self.infer(embedding, weights, mode=mode)
            return logits
        if mode == 'deploy':
            assert len(x) == 1 or len(x) == 4, \
                'in deploy mode, support extend and infer together or infer only'
            if len(x) == 1:
                embedding = x[0]
                N, _, h, w = embedding.size()
                logits = self.infer(embedding, weight=self.weight, mode=mode)
                return logits
            if len(x) == 4:
                embedding, Fs, Ys, labels = x
                # (num_class, c)
                self.weight.data = self.extend(Fs, Ys, labels, mode=mode).data
                # N, _, h, w = embedding.size()
                logits = self.infer(embedding, weight=self.weight, mode=mode)
                return logits


if __name__ == '__main__':
    N, way, shot, c, h, w = 4, 1, 5, 256, 41, 41
    num_class = 21
    embedding = torch.randn([N, c, h, w])
    Fs = torch.randn([N, way, shot, c, h, w])
    Ys = torch.randint(0, 2, size=[N, way, shot, h, w])
    cls = torch.randint(6, 10, size=[N, way])


    def run_GAM():
        gam = GAM(in_channels=256)
        tensor_Fs = torch.randn([4, 256, 41, 41])
        tensor_Ys = torch.randn([4, 41, 41])
        guided_Fs = gam(tensor_Fs, tensor_Ys)
        print(guided_Fs.size())


    def run_WeightEstimatorInitiate():
        awg = WeightEstimator(num_classes=21, depth=256)
        print(awg.weight.size())
        print(awg.weight.requires_grad)


    def run_WeightEstimatorTrainingModeForward():
        awg = WeightEstimator(num_classes=num_class, depth=c)
        logits = awg(embedding, Fs, Ys, cls, mode='training')
        print(logits.size())


    def run_WeightEstimatorDeployModeForward():
        awg = WeightEstimator(num_classes=num_class, depth=c)
        logits = awg(embedding, Fs, Ys, cls, mode='deploy')
        print(logits.size())
