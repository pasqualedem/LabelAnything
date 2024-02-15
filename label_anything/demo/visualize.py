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

from label_anything.experiment.utils import WrapperModule


colors = [
    # blue
    (0, 0, 255),
    # red
    (255, 0, 0),
    # green
    (0, 255, 0),
    # yellow
    (255, 255, 0),
    # purple
    (255, 0, 255),
    # cyan
    (0, 255, 255),
    # orange
    (255, 165, 0),
    # pink
    (255, 192, 203),
    # brown
    (139, 69, 19),
    # grey
    (128, 128, 128),
    # black
    (0, 0, 0),
    # white
    (255, 255, 255),
]

text_colors = [
    "blue",
    "red",
    "green",
    "yellow",
    "purple",
    "cyan",
    "orange",
    "pink",
    "brown",
    "grey",
    "black",
    "white",
]

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


def draw_points(img: Image, points: torch.Tensor, colors):
    img = np.array(img)
    
    for i, cat in enumerate(points):
        for point in cat:
            x, y = point
            x, y = int(x), int(y)
            img = cv2.circle(img, (x, y), 5, colors[i], -1)
    return img


def draw_masks(img: Image, masks: torch.Tensor, colors):
    # here masks is a dict having category names as keys
    # associated to a list of binary masks
    masked_image = resize(img.copy(), 256)
    
    for i, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                                np.asarray(colors[i], dtype="uint8"),
                                masked_image)
    
    masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(np.array(resize(img, 256)), 0.3, masked_image, 0.7, 0)


def draw_boxes(img: Image, boxes: torch.Tensor, colors):
    img = np.array(img)
    
    for i, cat in enumerate(boxes):
        for box in cat:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[i], 2)
    return img


def draw_seg(img: Image, seg: torch.Tensor, colors, num_classes, dims=None):
    resized_image = resize(img.copy(), seg.shape[-2:])
    masked_image = resized_image.copy()
    for i in range(1, num_classes):
        binary_mask = (seg == i)[0]
        mask = binary_mask.cpu().numpy()
        masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                                np.asarray(colors[i], dtype="uint8"),
                                masked_image)
    
    masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(np.array(resized_image), 0.6, masked_image, 0.4, 0)


def draw_all(img: Image, masks, boxes, points, colors):
    segmented_image = draw_masks(img, masks, colors)
    img = Image.fromarray(segmented_image)
    img = resize(img, 1024)
    img = draw_boxes(img, boxes, colors)
    img = Image.fromarray(img)
    img = draw_points(img, points, colors)
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
        unbatched["prompt_bboxes"][i],
        unbatched["prompt_points"][i],
        colors
    )
    for i in range(unbatched["images"].shape[0])
]
    plot_images(images, unbatched["classes"], dataset.categories["coco"])
    
    
def plot_segs(input, seg, gt, colors, dims):
    num_classes = len(input['classes'][0][0]) + 1
    image = get_image(input['images'][0, 0])
    segmask = draw_seg(
        image,
        seg.cpu(),
        colors,
        num_classes=num_classes
    )

    gtmask = draw_seg(
        image,
        gt,
        colors,
        num_classes=num_classes
    )
    blank_seg = Image.fromarray(np.zeros_like(segmask))
    blank_gt = Image.fromarray(np.zeros_like(gtmask))
    blank_segmask = draw_seg(
        blank_seg,
        seg.cpu(),
        colors,
        num_classes=num_classes
    )

    blank_gtmask = draw_seg(
        blank_gt,
        gt,
        colors,
        num_classes=num_classes
    )
    plots = [segmask, gtmask, blank_segmask, blank_gtmask, image, image]
    titles = ["Predicted", "Ground Truth", "Predicted", "Ground Truth", "Original", "Original"]

    subplots = plt.subplots(3, 2, figsize=(20, 30))
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
            if file_name == "pytorch_model.bin":
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