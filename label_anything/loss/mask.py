import torch
import torch.nn as nn

from einops import rearrange

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


class MaskEmbeddingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.1
        self.beta = 0.9
    
    def forward(self, result_dict):
        masks = result_dict[ResultDict.MASK_EMBEDDINGS]
        
        bg, fg = masks
        bg = rearrange(bg, "n b ... -> (b ...) n")
        fg = rearrange(fg, "n b ... -> (b ...) n")
        
        fg_sum = torch.nn.functional.softmax(fg.sum(dim=0))
        bg_sum = torch.nn.functional.softmax(bg.sum(dim=0))
        
        inters = self.alpha * ((calculate_entropy(bg).mean() + calculate_entropy(fg).mean()) / 2)
        union = self.beta * (((1 - calculate_entropy(fg_sum)) + (1 - calculate_entropy(bg_sum))) / 2)
        
        return {
            LossDict.VALUE: inters + union,
            LossDict.COMPONENTS: {
                "mask_int": inters.item(),
                "mask_uni": union.item(),   
            }
        } 
        