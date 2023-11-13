from pathlib import Path
import torch
import json
import pytest
from tqdm import tqdm

import time

from label_anything.data.examples import ExampleGeneratorPowerLawUniform

@pytest.mark.skip(reason="Not implemented")
def test_examples():
    DATA_DIR = Path.cwd() / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    IMAGES_DIR = RAW_DATA_DIR / "train2017"

    with open(RAW_DATA_DIR / "categories.json") as f:
        data = json.load(f)

    annotations = data["annotations"]
    id2category = {elem['id']: elem['name'] for elem in data['categories']}

    image_categories = {}
    category_images = {}

    for annotation in tqdm(annotations):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        if image_id not in image_categories:
            image_categories[image_id] = set()
        image_categories[image_id].add(category_id)
        if category_id not in category_images:
            category_images[category_id] = set()
        category_images[category_id].add(image_id)

    ALPHA = -1
    MIN_SIZE = 1

    TRIALS = 10
    N_EXAMPLES = 10
    TIMEIT = True

    example_generator = ExampleGeneratorPowerLawUniform(categories_to_imgs=category_images)

    if TIMEIT:
        start = time.time()
        for _ in range(TRIALS):
            query = annotations[torch.randint(0, len(annotations) - 1, (1,)).item()]
            query_categories = torch.tensor(list(set(image_categories[query['image_id']])), dtype=torch.int64)

            orig, sampled_images, sampled_classes = example_generator.generate_examples(query['image_id'], query_categories, N_EXAMPLES) 
        end = time.time()
        print(f"Time taken: {(end - start) / TRIALS}")
    else:
        for _ in range(TRIALS):
            print("==========================================================================")
            query = annotations[torch.randint(0, len(annotations) - 1, (1,)).item()]
            query_categories = torch.tensor(list(set(image_categories[query['image_id']])), dtype=torch.int64)
            print(f"Classes        : {[id2category[cat_id.item()] for cat_id in query_categories]}")
            print(f"Class ids: {query_categories.tolist()}")

            orig, sampled_images, sampled_classes = example_generator.generate_examples(query['image_id'], query_categories, N_EXAMPLES) 
            print(f"Original classes: {orig}")
            for cat_set in sampled_classes:
                print([id2category[cat_id] for cat_id in cat_set])

