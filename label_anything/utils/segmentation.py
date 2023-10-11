import numpy as np
import plotly.express as px


class ColorMap:
    def __init__(self):
        self.cmap = [
            "#000000",
            "#00ff00",
            "#ff0000",
            "#0000ff",
        ] + px.colors.qualitative.Alphabet
        self.cmap = [
            tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            for h in self.cmap
        ]

    def __getitem__(self, item):
        return self.cmap[item]


def tensor_to_segmentation_image(
    prediction, cmap: list = None, labels=None, return_clmap=False
) -> np.array:
    if cmap is None:
        cmap = ColorMap()
    if labels is None:
        labels = np.unique(prediction)
    segmented_image = np.ones((*prediction.shape, 3), dtype="uint8")
    for i in range(len(labels)):
        segmented_image[prediction == i] = cmap[i]
    if return_clmap:
        cmap = {labels[i]: cmap[i] for i in range(len(labels))}
        return segmented_image, cmap
    return segmented_image
