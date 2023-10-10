print("init")
import torch
import numpy as np
print("torch")
from label_anything.models import build_sam_vit_h, build_lam_vit_l, build_lam_vit_b
print("build")

@torch.no_grad()
def test_sam():
    sam = build_sam_vit_h()

    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    coords_torch = torch.as_tensor(input_point, dtype=torch.float)
    labels_torch = torch.as_tensor(input_label, dtype=torch.int)
    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

    box_torch = torch.as_tensor(input_box, dtype=torch.float)
    box_torch = box_torch[None, :]

    image = torch.rand(2, 3, 1024, 1024)
    mask = torch.rand(1, 1, 256, 256)

    points = (coords_torch, labels_torch)
    features = sam.image_encoder(image)

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=points,
        boxes=box_torch,
        masks=mask,
    )

    # Predict masks
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=features,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    print(low_res_masks.shape)


@torch.no_grad()
def test_lam():
    print("start")
    lam = build_lam_vit_b()
    print("lam")
    weights = torch.load("checkpoints/sam_vit_b_01ec64.pth") 
    lam.init_pretrained_weights(weights) 
    lam = lam.cuda()

    input_box_1 = np.array([[[425, 600, 700, 875]], [[125, 200, 300, 175]]])
    input_padding_1 = np.array([[1], [1]])
    input_point_1 = np.array([[[575, 750]], [[275, 350]]])
    input_label_1 = np.array([[0], [1]])

    input_box_2 = np.array([[[425, 600, 700, 875]], [[125, 200, 300, 175]]])
    input_padding_2 = np.array([[0], [1]])
    input_point_2 = np.array([[[575, 750]], [[275, 350]]])
    input_label_2 = np.array([[0], [1]])

    coords_torch = torch.as_tensor([input_point_1, input_point_2], dtype=torch.float)
    labels_torch = torch.as_tensor([input_label_1, input_label_2], dtype=torch.int)
    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    padding_torch = torch.as_tensor([input_padding_1, input_padding_2], dtype=torch.float)
    padding_torch = padding_torch[None, :, :]

    box_torch = torch.as_tensor([input_box_1, input_box_2], dtype=torch.float)
    box_torch = box_torch[None, :]

    images = torch.rand(1, 3, 3, 1024, 1024)
    masks = torch.rand(1, 2, 2, 256, 256)
    print("inputs")

    batch = {
        'target_image': images[:, 0].cuda(),
        'example_images': images[:, 1:].cuda(),
        'point_coords': coords_torch.cuda(),
        'point_labels': labels_torch.cuda(),
        'boxes': box_torch.cuda(),
        'box_flags': padding_torch.cuda(),
        'mask_inputs': masks.cuda(),
    }

    seg = lam(batch)
    print(seg.shape, "si cazzo")


test_lam()