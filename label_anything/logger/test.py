import comet_ml
from image_logger import Logger
import torch
import os
import random


def create_example_tensor(M, C, N):
    annotations_tensor = torch.zeros(M, len(C), N, 4)
    # Genera casualmente le coordinate delle bounding boxes per ciascuna annotazione
    for m in range(M):
        for c in C:
            for n in range(N):
                # Genera casualmente le coordinate [x1, y1, x2, y2] per la bounding box
                x1 = random.randint(0, 100)
                y1 = random.randint(0, 100)
                x2 = random.randint(x1 + 10, 500)
                y2 = random.randint(y1 + 10, 500)

                # Assegna le coordinate alla sequenza di quattro punti nel tensore
                annotations_tensor[m, c, n, 0] = x1
                annotations_tensor[m, c, n, 1] = y1
                annotations_tensor[m, c, n, 2] = x2
                annotations_tensor[m, c, n, 3] = y2
    return annotations_tensor


def convert_tensor_to_boxes(M, C, N, annotations_tensor: torch.Tensor):
    annotations = []
    # Itera attraverso gli esempi, classi e annotazioni nel tensore
    for m in range(M):
        for c in C:
            for n in range(N):
                # Estrai le coordinate della bounding box dal tensore
                boxes = annotations_tensor[m, c, n].tolist()

                # Crea un dizionario di annotazione nel formato richiesto
                annotation = {
                    "boxes": [boxes],
                    "label": str(c),
                    # "score": score, # Score
                }

                # Aggiungi questa annotazione alla lista delle annotazioni
                annotations.append(annotation)

    return annotations


if __name__ == "__main__":
    # experiment = comet_ml.Experiment(
    #     api_key=os.getenv("COMET_API_KEY"),
    #     project_name="IOLAB",
    # )
    # logger = Logger(experiment)
    M = 1  # Numero di esempi
    C = (0, 1)  # Numero di classi
    N = 3  # Numero massimo di annotazioni per classe
    annotations_tensor = create_example_tensor(M, C, N)
    annotations = convert_tensor_to_boxes(M, C, N, annotations_tensor)

    # Crea la struttura di annotazione finale
    annotations_data = [{"name": "Test", "data": annotations}]
    print(annotations_data)
    # Logga l'immagine con Comet.ml utilizzando la struttura di annotazione
    # logger.log_image("duck.jpeg", annotations=annotations_data)
