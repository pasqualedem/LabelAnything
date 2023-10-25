import os.path

import pandas as pd
import utils
import json
from argparse import ArgumentParser


def transform_bbox_to_coords(bbox):
    x, y, w, h = bbox
    return [x, y, x+w, y+h]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--instances_path')
    parser.add_argument('--out_path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.instances_path.endswith('.json'), "instances must be a json file."
    assert args.out_path.endswith('.json'), "data serialization supported just for json files."
    instances = utils.load_instances(args.instances_path)
    images = pd.DataFrame(instances['images'])
    annotations = pd.DataFrame(instances['annotations'])
    categories = pd.DataFrame(instances["categories"])

    images = images[["id", "coco_url"]]

    annotations["bbox"] = annotations["bbox"].map(transform_bbox_to_coords)

    categories = categories[["id", "name"]]

    categories = categories.to_dict(orient="records")
    images = images.to_dict(orient="records")
    annotations = annotations.to_dict(orient="records")

    instances = {
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    if os.path.exists(args.out_path):
        os.remove(args.out_path)

    with open(args.out_path, "w+") as out_file:
        json.dump(instances, out_file)
