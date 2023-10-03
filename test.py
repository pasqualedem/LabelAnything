import torch

from label_anything.models import build_sam_vit_h

sam = build_sam_vit_h()

coords = torch.tensor([[[10, 10], [20, 20], [30, 30]]])
labels = torch.tensor([[1, 1, 0]])
boxes = torch.tensor([[[10, 10, 20, 20], [20, 20, 30, 30], [30, 30, 40, 40]]])
image = torch.rand(1, 3, 1024, 1024)

points = (coords, labels)
features = sam.image_encoder(image)

sparse_embeddings, dense_embeddings = sam.prompt_encoder(
    points=points,
    boxes=boxes,
    masks=None,
)

# Predict masks
low_res_masks, iou_predictions = sam.mask_decoder(
    image_embeddings=features,
    image_pe=sam.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=False,
)
