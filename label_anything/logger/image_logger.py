import comet_ml
import torch


"""

'query_image': The input image as a torch tensor in Bx3xHxW format,
                already transformed for input to the model.
'example_images': The example images as a torch tensor in BxMx3xHxW format,
                already transformed for input to the model.
'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxMxCxNx2. Already transformed to the
                input frame of the model.
'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxMxCxN.
'boxes': (torch.Tensor) Batched box inputs, with shape BxMxCxNx4.
                Already transformed to the input frame of the model.
'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form BxMxCxHxW.
'box_flags': (torch.Tensor) Batched bounding box flags, with shape BxMxCxN.
"""


class Logger:
    def __init__(self, experiment):
        self.experiment = experiment

    def log_image(self, img_data, annotations):
        self.experiment.log_image(
            image_data=img_data,
            annotations=annotations,
        )

    def log_artifact(self, nome, annotation):
        self.experiment.log_artifact(nome, annotation)

    def log_metric(self, nome, metric):
        self.experiment.log_metric(nome, metric)

    def structure_annotations(annotations):
        return [{"name": "Predictions", "data": annotations}]

    def __data_to_single__(data: dict) -> dict[torch.Tensor]:
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

    def get_shape_tensor(tensor: torch.Tensor):
        M = tensor.shape[0]
        C = tensor.shape[1]
        N = tensor.shape[2]
        return M, C, N


if "__main__" == __name__:
    import os
    import random

    experiment = comet_ml.Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="IOLAB",
    )
    logger = Logger(experiment)

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

    M = 1  # Numero di esempi
    C = (0, 1)  # Numero di classi
    N = 3  # Numero massimo di annotazioni per classe
    annotations_tensor = create_example_tensor(M, C, N)

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

    # Crea la struttura di annotazione finale
    annotations_data = [{"name": "Test", "data": annotations}]
    print(annotations_data)
    # Logga l'immagine con Comet.ml utilizzando la struttura di annotazione
    logger.log_image(
        "/home/emanuele/Workspace/dottorato/LabelAnything/duck.png",
        annotations=annotations_data,
    )


"""
Example usage:
==============================================================
image:
- a path (string) to an image
- file-like containg image
- numpy matrix
- tensorflow tensor
- pytorch tensor
- list of tuple of values
- PIL image
$ logger.log_image("image_name", image)
==============================================================
annotation = {"boxes": [x, y, w, h]}
annotation = [
    {
        "label": 'ciola',
        "boxes": [0, 0, 100, 100],
    },
    {
        "label": 'figa'
        "boxes": [0, 0, 100, 101],
    }
]
$ logger.log_artifact("annotation_name", annotation)
==============================================================
punto = {
    "x": 100,
    "y": 200,
}
punti = [
    {
        "x": 100,
        "y": 200,
    },
    {
        "x": 200,
        "y": 300,
    },
]
$ logger.log_artifact("punti_name", punto)
==============================================================
maschera = {
    "points": [
        [100, 200],
        [200, 300],
        [300, 200],
        [200, 100],
    ],
}
maschere = [
    {
        "points": [
            [100, 200],
            [200, 300],
            [300, 200],
            [200, 100],
        ],
    },
    {
        "points": [
            [400, 500],
            [500, 600],
            [600, 500],
            [500, 400],
        ],
    },
]
$ logger.log_artifact("maschera", maschera)
==============================================================
"""
