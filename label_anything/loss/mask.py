import torch
import torch.nn as nn

from einops import rearrange

from label_anything.loss.utils import loss_orthogonality
from label_anything.utils.utils import LossDict, ResultDict

def calculate_entropy(probabilities):
    """
    Calculate the entropy of a probability distribution.

    Args:
        probabilities (torch.Tensor): A tensor containing the probabilities. 
                                      Must be a valid probability distribution 
                                      (non-negative, sums to 1 along the desired dimension).

    Returns:
        torch.Tensor: The entropy of the distribution.
    """
    # Ensure numerical stability by adding a small epsilon
    epsilon = 1e-10
    probabilities = probabilities + epsilon
    
    # Compute the entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities) / torch.log(torch.tensor(2.0)), dim=-1)
    return entropy


def loss_balance(mask, tol=0.25):
    B, N, _, H, W = mask.shape
    eps = 1e-6
    
    sum_mask = mask.view(mask.size(0), N, -1).sum(dim=-1)
    target = (torch.abs(sum_mask).sum(dim=1) / N).unsqueeze(-1)
    balance_loss = ((torch.abs(sum_mask - target) / (target + eps))).sum(dim=1) / N
    balance_loss = nn.functional.relu(balance_loss - tol)
    
    return balance_loss.sum() / B


import torch

class MaskEmbeddingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.20
        self.beta = 0.40
        self.gamma = 0.40
    
    def forward(self, result_dict):
        masks = result_dict[ResultDict.MASK_EMBEDDINGS]
        
        bg, fg = masks
        bg = rearrange(bg, "n b ... -> b n ...")
        fg = rearrange(fg, "n b ... -> b n ...")
        
        balance_bg = loss_balance(bg)
        balance_fg = loss_balance(fg)
        balance = ((balance_bg + balance_fg) / 2) * self.alpha
        
        ortho_fg = loss_orthogonality(fg)
        ortho_bg = loss_orthogonality(bg)
        ortho = ((ortho_bg + ortho_fg) / 2) * self.beta
        
        fg = rearrange(fg, "b n ... -> (b ...) n")
        bg = rearrange(bg, "b n ... -> (b ...) n")
        
        entropy_fg = calculate_entropy(fg).mean()
        entropy_bg = calculate_entropy(bg).mean()
        entropy = ((entropy_bg + entropy_fg) / 2) * self.gamma  
        
        return {
            LossDict.VALUE: ortho + balance + entropy,
            LossDict.COMPONENTS: {
                "balance": balance.item(),
                "ortho": ortho.item(),   
                "entropy": entropy.item()
            }
        } 
        