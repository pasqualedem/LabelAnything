import cv2
import numpy as np
from PIL import Image, ImageDraw
import colorsys


def generate_class_colors(num_classes):
    # Genera automaticamente una lista di colori in base al numero di classi
    class_colors = []
    for i in range(num_classes):
        # Usa una logica per generare colori diversi in base all'indice
        hue = (i / num_classes) * 360  # Distribuzione uniforme nel cerchio dei colori
        rgb_color = colorsys.hsv_to_rgb(hue / 360, 1, 1)
        color = tuple(int(c * 255) for c in rgb_color)
        class_colors.append(color)
    return class_colors


def image_with_points(query_image, points_tensor):
    # Converte il tensore in un'immagine Pillow
    image = query_image.permute(1, 2, 0).cpu().numpy()
    image_pil = Image.fromarray((image * 255).astype("uint8"))

    # Estrae le dimensioni del tensore dei punti
    num_classes, num_examples, _ = points_tensor.size()

    # Genera una lista di colori in base al numero di classi
    class_colors = generate_class_colors(num_classes)

    # Crea un oggetto ImageDraw per disegnare i punti
    draw = ImageDraw.Draw(image_pil)

    # Disegna i punti sull'immagine per ciascuna classe
    for class_idx in range(num_classes):
        class_color = class_colors[class_idx]

        for example_idx in range(num_examples):
            x, y = points_tensor[class_idx, example_idx]
            if x == -1 and y == -1:
                # Ignora i punti con valori di -1
                continue

            # Usa il colore specifico per ciascuna classe
            color = class_color

            # Disegna il punto sull'immagine con Pillow
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

    return image_pil


def structure_annotations(annotations):
    structured_annoations = [
        {
            "name": "Predictions",
            "data": [
                {
                    "points": [annotations],
                    "label": "1",
                    "score": 0.9,
                }
            ],
        }
    ]
    return structured_annoations


def image_with_points(query_image, points_tensor, label):
    # Converti il tensore in un'immagine numpy
    image = query_image.cpu().numpy().transpose(1, 2, 0)

    # Estrai i punti dall'array dei tensori
    points = points_tensor.cpu().numpy().astype(np.int)

    # Copia l'immagine originale per evitare di modificarla direttamente
    image_with_overlay = np.copy(image)

    # Disegna i punti sull'immagine
    for point in points:
        x, y = point
        # Usa un cerchio rosso per rappresentare il punto
        cv2.circle(image_with_overlay, (x, y), 3, (0, 0, 255), -1)

    # Inserisci l'etichetta sull'immagine
    cv2.putText(
        image_with_overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )
    return image_with_overlay


def extract_labels_and_points_from_tensor(tensor):
    """
    Args:
        tensor (torch.Tensor): tensor to extract labels and points

    Returns:
        list[dict]: list of dictionaries, each containing label and points
    """
    annotations = []

    for i in range(tensor.size(0)):
        label = str(i)
        points = (tensor[i] > 0).nonzero()  # Trova gli indici dei punti non nulli

        annotation = {
            "label": label,
            "points": points.tolist(),
        }
        annotations.append(annotation)

    return annotations


def extract_vertices_from_tensor(tensor):
    """
    Args:
        binary_tensor (torch.Tensor()): tensor to extract vertices

    Returns:
        list[int]: list of indices of vertices
    """
    binary_array = (tensor.cpu().numpy() * 255).astype(np.uint8)

    # Trova i contorni nella figura
    contours, _ = cv2.findContours(
        binary_array,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Estrai i vertici dalla lista dei contorni e convertili in un formato desiderato
    vertices_list = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = approx.squeeze().tolist()
        flattened_vertices = [coord for point in vertices for coord in point]
        vertices_list.extend(flattened_vertices)

    return vertices_list


def data_to_single(data: dict):
    output = {}
    for key in data:
        output[key] = data[key]

    # Estrae i tensori dal dizionario output
    (
        query_image,
        example_images,
        point_coords,
        point_labels,
        boxes,
        mask_inputs,
        box_flags,
    ) = output.values()

    return (
        query_image,
        example_images,
        point_coords,
        point_labels,
        boxes,
        mask_inputs,
        box_flags,
    )


def extract_boxes_from_tensor(tensor):
    annotations = []

    for i in range(tensor.size(0)):
        label = str(i)
        boxes = tensor[i, :, 1:].tolist()

        for box in boxes:
            annotation = {
                "label": label,
                "boxes": [box],
            }
            annotations.append(annotation)

    return annotations