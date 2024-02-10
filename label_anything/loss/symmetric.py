import torch
from torch.nn.functional import normalize, cross_entropy


class SymmetricLoss(torch.nn.Module):
    def __init__(
            self,
            t: float = 1.0,
            norm: bool = True,
    ):
        super().__init__()
        self.t = torch.exp(torch.as_tensor([t]).float())
        self.norm = norm

    def forward(self, input1, input2, labels):
        input1 = normalize(input1, p=2, dim=1) if self.norm else input1
        input2 = normalize(input2, p=2, dim=1) if self.norm else input2
        logits = torch.mm(input1, input2.t()) * self.t.to(input1.device)
        loss1 = cross_entropy(logits, labels)
        loss2 = cross_entropy(logits.t(), labels.t())
        return (loss1 + loss2) / 2
