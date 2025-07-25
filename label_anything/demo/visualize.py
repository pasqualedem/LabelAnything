import wandb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from torchvision.transforms.functional import resize
from safetensors.torch import load_file
from einops import rearrange
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import cv2

from label_anything.data import utils
from label_anything.data.utils import AnnFileKeys, PromptType, BatchKeys, flags_merge
from accelerate import Accelerator

from label_anything.demo.utils import COLORS
from label_anything.experiment.utils import WrapperModule
from label_anything.logger.utils import take_image


def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(b, device) for b in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch

def obtain_batch(superdatset, dataset, images, image_ids, cat_ids, classes, img_sizes, image_key, prompts, ground_truths=None):
    bboxes, masks, points = prompts
    # obtain padded tensors
    bboxes, flag_bboxes = dataset.annotations_to_tensor(
        bboxes, img_sizes, PromptType.BBOX
    )
    masks, flag_masks = dataset.annotations_to_tensor(
        masks, img_sizes, PromptType.MASK
    )
    points, flag_points = dataset.annotations_to_tensor(
        points, img_sizes, PromptType.POINT
    )

    # obtain ground truths
    if ground_truths is None:
        ground_truths = dataset.compute_ground_truths(image_ids, cat_ids)

    # stack ground truths
    dims = torch.tensor(img_sizes)
    max_dims = torch.max(dims, 0).values.tolist()
    ground_truths = torch.stack(
        [utils.collate_gts(x, max_dims) for x in ground_truths]
    )

    if dataset.load_gts:
        # convert the ground truths to the right format
        # by assigning 0 to n-1 to the classes
        ground_truths_copy = ground_truths.clone()
        # set ground_truths to all 0s
        ground_truths = torch.zeros_like(ground_truths)
        for i, cat_id in enumerate(cat_ids):
            if cat_id == -1:
                continue
            ground_truths[ground_truths_copy == cat_id] = i
            
    flag_examples = flags_merge(flag_masks, flag_points, flag_bboxes)

    data_dict = {
        image_key: images,
        BatchKeys.PROMPT_MASKS: masks,
        BatchKeys.FLAG_MASKS: flag_masks,
        BatchKeys.PROMPT_POINTS: points,
        BatchKeys.FLAG_POINTS: flag_points,
        BatchKeys.PROMPT_BBOXES: bboxes,
        BatchKeys.FLAG_BBOXES: flag_bboxes,
        BatchKeys.FLAG_EXAMPLES: flag_examples,
        BatchKeys.DIMS: dims,
        BatchKeys.CLASSES: classes,
        BatchKeys.IMAGE_IDS: image_ids,
        BatchKeys.GROUND_TRUTHS: ground_truths,
    }
    batch = superdatset.collate_fn([(data_dict, "coco")])
    return batch


def get_image(image_tensor):
    MEAN = np.array([123.675, 116.280, 103.530]) / 255
    STD = np.array([58.395, 57.120, 57.375]) / 255
    unnormalized_image = (image_tensor.cpu().numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    return Image.fromarray(unnormalized_image)


def draw_points(img: Image, points: torch.Tensor, colors, flag_points=None):
    img = np.array(img)
    
    if flag_points is None:
        flag_points = torch.ones_like(points, dtype=torch.bool)[:, :, 0]
    
    for i, (cat, flag_cat) in enumerate(zip(points, flag_points)):
        for point, flag in zip(cat, flag_cat):
            if not flag:
                continue
            x, y = point
            x, y = int(x), int(y)
            img = cv2.circle(img, (x, y), 5, colors[i], -1)
    return img


def draw_masks(img: Image, masks: torch.Tensor, colors, flag_masks=None):
    # here masks is a dict having category names as keys
    # associated to a list of binary masks
    orig_shape = img.size
    masked_image = resize(img.copy(), 256)
    
    for i, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                                np.asarray(colors[i], dtype="uint8"),
                                masked_image)
    
    masked_image = masked_image.astype(np.uint8)
    overlap = cv2.addWeighted(np.array(resize(img, 256)), 0.3, masked_image, 0.7, 0)
    return resize(Image.fromarray(overlap), orig_shape)


def draw_boxes(img: Image, boxes: torch.Tensor, colors, flag_bboxes=None):
    img = np.array(img)

    if flag_bboxes is None:
        flag_bboxes = torch.ones_like(boxes, dtype=torch.bool)[:, :, 0]
    
    for i, (cat, flag_cat) in enumerate(zip(boxes, flag_bboxes)):
        for box, flag in zip(cat, flag_cat):
            if not flag:
                continue
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[i], 2)
    return img


def draw_seg(img: Image, seg: torch.Tensor, colors, num_classes):
    masked_image = img.copy()
    for i in range(1, num_classes):
        binary_mask = (seg == i)[0]
        mask = binary_mask.cpu().numpy()
        masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                                np.asarray(colors[i], dtype="uint8"),
                                masked_image)
    
    masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(np.array(img), 0.6, masked_image, 0.4, 0)


def draw_all(img: Image, masks, boxes, points, colors, flag_masks=None, flag_bboxes=None, flag_points=None):
    
    img = draw_masks(img, masks, colors, flag_masks=flag_masks)
    img = draw_boxes(img, boxes, colors, flag_bboxes=flag_bboxes)
    img = Image.fromarray(img)
    img = draw_points(img, points, colors, flag_points=flag_points)
    img = Image.fromarray(img)
    return img


def plot_images(images, classes, ids_to_cats):
    num_images = len(images)
    cols = 2
    rows = (num_images + 1) // cols

    fig, ax = plt.subplots(rows, cols, figsize=(20, 20))

    for i, img in enumerate(images):
        cats = [ids_to_cats[cat]["name"] for cat in classes[i]]
        if rows == 1:
            ax[i % cols].imshow(img)
            ax[i % cols].set_title(", ".join(cats))
            ax[i % cols].axis("off")
        else:
            ax[i // cols, i % cols].imshow(img)
            ax[i // cols, i % cols].set_title(", ".join(cats))
            ax[i // cols, i % cols].axis("off")
        
    plt.show()


def plot_all(dataset, batch, colors):
    unbatched = {k : v[0] for k, v in batch.items()}
    images = [
    draw_all(
        get_image(unbatched["images"][i]),
        unbatched["prompt_masks"][i],
        unbatched["prompt_bboxes"][i][unbatched["flag_bboxes"][i]],
        unbatched["prompt_points"][i][unbatched["flag_points"][i]],
        colors
    )
    for i in range(unbatched["images"].shape[0])
]
    plot_images(images, unbatched["classes"], dataset.categories["coco"])
    
    
def unnormalize(image_tensor):
    MEAN = torch.tensor([123.675, 116.280, 103.530], device=image_tensor.device) / 255
    STD = torch.tensor([58.395, 57.120, 57.375], device=image_tensor.device) / 255
    unnormalized_image = (image_tensor * STD[:, None, None]) + MEAN[:, None, None]
    unnormalized_image = (unnormalized_image * 255).byte()
    return unnormalized_image


def create_rgb_segmentation(segmentation, num_classes=None):
    """
    Convert a segmentation map to an RGB visualization using a precise colormap.

    Args:
        segmentation (torch.Tensor): Segmentation map of shape [B, H, W] where
                                      each pixel contains class labels (natural numbers).
        num_classes (int): The number of unique classes in the segmentation.

    Returns:
        torch.Tensor: RGB visualization of shape [B, 3, H, W].
    """
    if len(segmentation.shape) == 4:
        segmentation = segmentation.argmax(dim=1)
    if num_classes is None:
        num_classes = segmentation.max().item() + 1
    
    # Define a precise colormap for specific classes
    colormap = torch.tensor(COLORS, dtype=torch.uint8)  # Ensure dtype is uint8

    # Initialize an empty tensor for RGB output
    B, H, W = segmentation.shape
    rgb_segmentation = torch.zeros((B, 3, H, W), dtype=torch.uint8)

    # Loop through each class and assign the corresponding RGB color
    for class_id in range(num_classes):
        # Create a mask for the current class
        class_mask = (segmentation == class_id).unsqueeze(1)  # Shape: [B, 1, H, W]
        # Assign the corresponding color to the rgb_segmentation
        rgb_segmentation += class_mask * colormap[class_id].view(1, 3, 1, 1)  # Broadcasting

    return rgb_segmentation
    
    
def plot_seg_gt(input, seg, gt, colors, dims, classes, custom_preprocess=False, resize_to_model_shape=True):
    query_dim = dims[0, 0]
    num_classes = len(classes) + 1
    seg = seg.cpu()
    dims = dims.cpu()
    image = unnormalize(input["images"][0, 0]).cpu()
    model_image_size = image.shape[-2:]
    if custom_preprocess:
        image = take_image(image, dims=query_dim, input_shape=1024)
    else:
        image = resize(image, (query_dim[0], query_dim[1]))
        seg = seg[:, : query_dim[0], : query_dim[1]]
        gt = gt[:, : query_dim[0], : query_dim[1]]
    rgb_seg = create_rgb_segmentation(seg, num_classes=num_classes+1)[0]
    rgb_gt = create_rgb_segmentation(gt, num_classes=num_classes+1)[0]
    
    blended_image_pred = image.clone()
    blended_image_pred[rgb_seg > 0] = rgb_seg[rgb_seg > 0]
    
    blended_image_gt = image.clone()
    blended_image_gt[rgb_gt > 0] = rgb_gt[rgb_gt > 0]
    
    plots = [image, rgb_gt, blended_image_gt, rgb_seg, blended_image_pred]
    
    if resize_to_model_shape:
        for i in range(len(plots)):
            plots[i] = resize(plots[i], model_image_size)
    titles = [
        "Image",
        "Ground Truth",
        "Blended GT",
        "Pred",
        "Blended Pred",
    ]
    return plots, titles


def plot_seg(input, seg, colors, dims, classes):
    query_dim = dims[0, 0]
    num_classes = len(classes) + 1
    query_image = input["images"][0, 0]
    image = get_image(take_image(query_image, dims=query_dim, input_shape=query_image.shape[-1]))
    print(image)
    seg = seg[:, : query_dim[0], : query_dim[1]]
    segmask = draw_seg(image, seg.cpu(), colors, num_classes=num_classes)
    blank_seg = Image.fromarray(np.zeros_like(segmask))
    blank_segmask = draw_seg(
        blank_seg, seg.cpu(), colors, num_classes=num_classes
    )
    plots = [segmask, blank_segmask, image, image]
    titles = [
        "Overlay",
        "Mask",
    ]

    subplots = plt.subplots(1, 2, figsize=(20, 30))
    for i, (plot, title) in enumerate(zip(plots, titles)):
        subplots[1].flatten()[i].imshow(plot)
        subplots[1].flatten()[i].set_title(title)
        subplots[1].flatten()[i].axis("off")
    return plots, titles
        
        
def resize_ground_truth(ground_truth, dims):
    return ground_truth[:dims[0], :dims[1]]


def load_from_wandb(run_id, wandb_folder):
    api = wandb.Api()
    run = api.run(f"cilabuniba/LabelAnything/{run_id}")
    files = run.files()
    model_file = None
    config_file = None
    for file in files:
        # Extract name from path
        file_name = file.name.split("/")[-1]
        if "model" in file.name and wandb_folder in file.name:
            if file_name == "pytorch_model.bin" or file_name == "model.safetensors":
                model_file = file.download(replace=True, root="streamlit")
        if file_name == "config.yaml":
            config_file = file.download(replace=True, root="streamlit")
    return model_file.name, config_file.name
    
    
def get_embeddings_names(batch, embeddings_dir):
    image_ids = batch['image_ids']
    z_filled = [[str(image_id).zfill(12) for image_id in item] for item in image_ids]
    flattened = [item for sublist in z_filled for item in sublist]
    # filter embeddings already present in EMBEDDINGS_DIR
    filtered = [item for item in flattened if not Path(f"{embeddings_dir}/{item}.safetensors").exists()]
    names = " ".join(filtered)
    print(names)
    return names
    
    
def set_embeddings(accelerator, batch, embeddings_dir):
    item = batch['image_ids'][0]
    safetensors = []
    for id in item:
        safetensors.append(load_file(f"{embeddings_dir}/{str(id).zfill(12)}.safetensors")['embedding'])
        
    safetensors = torch.stack(safetensors).unsqueeze(0)
    safetensors = safetensors.to(accelerator.device)
    batch['embeddings'] = safetensors
    return batch


def reduce_embeddings(examples_class_embeddings):
    b, n, c, d = examples_class_embeddings.shape
    embeddings = rearrange(examples_class_embeddings, 'b n c d -> (b n c) d')

    # Perform t-SNE dimensionality reduction
    perplexity = min(10, n*c - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())
    return embeddings_2d


def plot_emebddings(examples_class_embeddings, example_flags, text_colors):
    b, n, c, d = examples_class_embeddings.shape
    embeddings_2d = reduce_embeddings(examples_class_embeddings)
    flags = rearrange(example_flags, 'b n c -> (b n c)')

    # Plot the 2D embeddings grouped per class
    fig = plt.figure(figsize=(8, 8))
    for i in range(n):
        for j in range(c):
            x, y = embeddings_2d[i * c + j]
            plt.scatter(x, y, color=text_colors[j], label=f"Class {j}", s=50)
            
    for i in range(n):
        for j in range(c):
            x, y = embeddings_2d[i * c + j]
            if flags[i * c + j]:
                plt.text(x, y, f"{i}-{j}", fontsize=12)
            else:
                plt.text(x, y, f"{i}-{j}", fontsize=12, color='red')


    fig.suptitle('t-SNE Visualization of Embeddings')
    fig.show()
    return fig