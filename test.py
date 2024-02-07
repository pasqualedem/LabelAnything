from label_anything.data.weedmap import WeedMapTestDataset
from label_anything.data.transforms import CustomResize, CustomNormalize
from torchvision.transforms import Compose, PILToTensor
import numpy as np
import lovely_tensors as lt
lt.monkey_patch()


preprocess = Compose([
    CustomResize(1024),
    PILToTensor(),
    CustomNormalize()
])


weed = WeedMapTestDataset(
    train_root="/ext/stalla/LabelAnything/WeedMap/0_rotations_processed_003_test/RedEdge/000",
    test_root="/ext/stalla/LabelAnything/WeedMap/0_rotations_processed_003_test/RedEdge/003",
    transform=preprocess,
)

print(weed.extract_prompts())

print(weed.__getitem__(0))