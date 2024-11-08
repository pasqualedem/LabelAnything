print("init")
import torch
import numpy as np
import pytest

print("torch")
from label_anything.models import (
    build_lam_vit_h,
    build_lam_vit_l,
    build_lam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    build_lam_no_vit,
)

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
    lam = build_lam_no_vit()
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
    padding_torch = torch.as_tensor(
        [input_padding_1, input_padding_2], dtype=torch.float
    )
    padding_torch = padding_torch[None, :, :]

    box_torch = torch.as_tensor([input_box_1, input_box_2], dtype=torch.float)
    box_torch = box_torch[None, :]

    images = torch.rand(1, 3, 3, 1024, 1024)
    masks = torch.rand(1, 2, 2, 256, 256)
    mask_flags = torch.randint(0, 2, (1, 2, 2)).bool()
    print("inputs")

    batch = {
        "images": images.cuda(),
        "prompt_points": coords_torch.cuda(),
        "flags_points": labels_torch.cuda(),
        "prompt_bboxes": box_torch.cuda(),
        "flags_bboxes": padding_torch.cuda(),
        "prompt_masks": masks.cuda(),
        "flags_masks": mask_flags.cuda(),
        "dims": torch.tensor([[[480, 640], [480, 640], [480, 640]]]).cuda(),
    }

    seg = lam(batch)
    print(seg.shape, "seg.shape")


@pytest.mark.lam
def test_embedding():
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
    padding_torch = torch.as_tensor(
        [input_padding_1, input_padding_2], dtype=torch.float
    )
    padding_torch = padding_torch[None, :, :]

    box_torch = torch.as_tensor([input_box_1, input_box_2], dtype=torch.float)
    box_torch = box_torch[None, :]

    embeddings = torch.rand(1, 3, 256, 64, 64)
    masks = torch.rand(1, 2, 2, 256, 256)
    mask_flags = torch.randint(0, 2, (1, 2, 2)).bool()
    print("inputs")

    batch = {
        "embeddings": embeddings.cuda(),
        "prompt_points": coords_torch.cuda(),
        "flag_points": labels_torch.cuda(),
        "prompt_bboxes": box_torch.cuda(),
        "flag_bboxes": padding_torch.cuda(),
        "prompt_masks": masks.cuda(),
        "flag_masks": mask_flags.cuda(),
        "dims": torch.tensor([[[480, 640], [480, 640], [480, 640]]]).cuda(),
    }

    seg = lam(batch)
    print(seg.shape, "seg.shape")
    assert seg.shape == (1, 2, 480, 640)


@pytest.mark.lam
def test_embedding_predict():
    print("start")
    lam = build_lam_vit_b()
    print("lam")
    weights = torch.load("checkpoints/sam_vit_b_01ec64.pth")
    lam.init_pretrained_weights(weights)
    lam = lam.cuda()


    embeddings = torch.rand(1, 256, 64, 64).cuda()
    class_embeddings = torch.rand(1, 5, 256).cuda()
    
    print("inputs")

    batch = {
        "embeddings": embeddings.cuda(),
        "dims": torch.tensor([[[480, 640], [480, 640], [480, 640]]]).cuda(),
    }

    seg = lam.predict(batch, class_embeddings)
    print(seg.shape, "seg.shape")
    assert seg.shape == (1, 5, 480, 640)