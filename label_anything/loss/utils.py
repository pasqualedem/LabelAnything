import torch

from label_anything.utils.utils import substitute_values


def get_reduction(reduction: str):
    if reduction == "none":
        return lambda x: x
    elif reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    
    
def get_weight_matrix_from_labels(labels, num_classes, ignore_index=-100):  
    there_is_ignore = ignore_index in labels
    if there_is_ignore:
        weight_labels = labels.clone()
        weight_labels += 1
        weight_labels[weight_labels == ignore_index + 1] = 0
        weight_num_classes = num_classes + 1
    else:
        weight_labels = labels
        weight_num_classes = num_classes
    weights = torch.ones(weight_num_classes, device=labels.device)
    classes, counts = weight_labels.unique(return_counts=True)
    classes = classes.long()
    if there_is_ignore:
        weights[classes] = 1 / torch.log(1.1 + counts / counts.sum())
        weights[0] = 0
        class_weights = weights[1:]
    else:
        weights[classes] = 1 / torch.log(1.1 + counts / counts.sum())
        class_weights = weights
    wtarget = substitute_values(
        weight_labels,
        weights,
        unique=torch.arange(weight_num_classes, device=labels.device),
    )
    return wtarget, class_weights


def loss_orthogonality(embedding):
    B, N = embedding.shape[:2]
    # Appiattisci la maschera per ogni prototipo in un vettore di dimensione (B, N, H*W)
    emb_flat = embedding.view(embedding.size(0), N, -1)  # BxNx(H*W)

    # Normalizza le maschere lungo l'asse H*W
    norm_emb = torch.norm(emb_flat, p=2, dim=-1, keepdim=True)  # BxNx1
    emb_flat_normalized = emb_flat / (norm_emb + 1e-8)  # BxNx(H*W) (evita la divisione per 0)

    # Calcola il prodotto scalare tra tutte le maschere (cross-prodotto tra le maschere)
    # La matrice risultante ha dimensione (B, N, N)
    similarity_matrix = torch.bmm(emb_flat_normalized, emb_flat_normalized.transpose(1, 2))  # BxNxN

    # Imposta gli elementi diagonali (prodotto della maschera con se stessa) a 0, perché non vogliamo penalizzare l'auto-similitudine
    mask_eye = torch.eye(N, device=embedding.device).unsqueeze(0).expand(embedding.size(0), -1, -1)  # BxNxN
    similarity_matrix = similarity_matrix * (1 - mask_eye)  # Rimuovi i termini diagonali

    # Calcola la penalità di ortogonalità come la somma dei valori assoluti fuori diagonale
    orthogonality_loss = torch.abs(similarity_matrix).sum() / (B * (N**2 - N))

    return orthogonality_loss