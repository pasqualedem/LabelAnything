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


def loss_balance(mask):
    B, N, _, H, W = mask.shape
    
    # Calcola la somma di ogni maschera lungo H e W (per ciascun prototipo)
    sum_mask = mask.view(mask.size(0), N, -1).sum(dim=-1)  # BxN
    
    # Calcola il valore medio per ogni maschera (H * W) / N
    target = (H * W) / N
    
    # Calcola la perdita di bilanciamento come la somma della differenza assoluta
    balance_loss = torch.abs(sum_mask - target).sum()  # Somma su tutto il batch e i prototipi
    
    return balance_loss


import torch

def loss_orthogonality(mask):
    B, N, _, H, W = mask.shape
    # Appiattisci la maschera per ogni prototipo in un vettore di dimensione (B, N, H*W)
    mask_flat = mask.view(mask.size(0), N, -1)  # BxNx(H*W)
    
    # Normalizza le maschere lungo l'asse H*W
    norm_mask = torch.norm(mask_flat, p=2, dim=-1, keepdim=True)  # BxNx1
    mask_flat_normalized = mask_flat / (norm_mask + 1e-8)  # BxNx(H*W) (evita la divisione per 0)
    
    # Calcola il prodotto scalare tra tutte le maschere (cross-prodotto tra le maschere)
    # La matrice risultante ha dimensione (B, N, N)
    similarity_matrix = torch.bmm(mask_flat_normalized, mask_flat_normalized.transpose(1, 2))  # BxNxN
    
    # Imposta gli elementi diagonali (prodotto della maschera con se stessa) a 0, perché non vogliamo penalizzare l'auto-similitudine
    mask_eye = torch.eye(N, device=mask.device).unsqueeze(0).expand(mask.size(0), -1, -1)  # BxNxN
    similarity_matrix = similarity_matrix * (1 - mask_eye)  # Rimuovi i termini diagonali
    
    # Calcola la penalità di ortogonalità come la somma dei valori assoluti fuori diagonale
    orthogonality_loss = torch.abs(similarity_matrix).sum()  # Somma su tutte le coppie di maschere
    
    return orthogonality_loss



class MaskEmbeddingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.1
        self.beta = 0.9
    
    def forward(self, result_dict):
        masks = result_dict[ResultDict.MASK_EMBEDDINGS]
        
        bg, fg = masks
        bg = rearrange(bg, "n b ... -> b n ...")
        fg = rearrange(fg, "n b ... -> b n ...")
        
        balance_bg = loss_balance(bg)
        balance_fg = loss_balance(fg)
        balance = (balance_bg + balance_fg) / 2
        
        ortho_fg = loss_orthogonality(fg)
        ortho_bg = loss_orthogonality(bg)
        ortho = (ortho_bg + ortho_fg) / 2
        
        
        return {
            LossDict.VALUE: ortho + balance,
            LossDict.COMPONENTS: {
                "balance": balance.item(),
                "ortho": ortho.item(),   
            }
        } 
        