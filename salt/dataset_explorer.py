import copy
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from distinctipy import distinctipy
from PIL import Image
from pycocotools import mask as mask_utils


def init_dataset(dataset_folder, image_paths, categories, dataset_json_path):
    dataset_json = {
        "categories": categories,
        "images": {},
    }
    for image_path in image_paths:
        im = Image.open(Path(dataset_folder) / image_path)
        dataset_json["images"][str(image_path)] = {
            "width": im.size[0],
            "height": im.size[1],
            "annotations": [],
        }
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)


def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    return list(itertools.chain(*coords))


def parse_mask_to_coco(image_id, anno_id, image_mask, category_id):
    start_anno_id = anno_id
    fortran_binary_mask = np.asfortranarray(image_mask)
    encoded_mask = mask_utils.encode(fortran_binary_mask)
    x, y, width, height = mask_utils.toBbox(encoded_mask)
    annotation = {
        "id": start_anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": encoded_mask,
    }
    annotation["segmentation"]["counts"] = str(
        annotation["segmentation"]["counts"], "utf-8"
    )
    return annotation


class DatasetExplorer:
    def __init__(self, dataset_folder, categories=None, dataset_json_path=None):
        self.dataset_folder = Path(dataset_folder)
        self.image_paths = [
            str(image.relative_to(self.dataset_folder))
            for image in (self.dataset_folder / "images").iterdir()
            if image.suffix in (".jpg", ".png")
        ]
        self.dataset_json_path = Path(dataset_json_path)
        if not self.dataset_json_path.exists():
            self.__init_dataset_json(categories)
        with open(dataset_json_path, "r") as f:
            self.dataset = json.load(f)

        self.categories = self.dataset["categories"]
        self.global_annotation_id = 0
        for image_info in self.dataset["images"].values():
            self.global_annotation_id += len(image_info["annotations"])

        self.category_colors = distinctipy.get_colors(len(self.categories), rng=len(self.categories))
        self.category_colors = [
            tuple([int(255 * c) for c in color]) for color in self.category_colors
        ]

    def __init_dataset_json(self, categories):
        if not categories:
            raise ValueError("No categories provided")
        init_dataset(
            self.dataset_folder, self.image_paths, categories, self.dataset_json_path
        )

    def get_colors(self, category_id):
        return self.category_colors[category_id]

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def get_num_images(self):
        return len(self.image_paths)

    def get_image_data(self, image_id):
        image_name = self.image_paths[image_id]
        image_path = self.dataset_folder / image_name
        image = cv2.imread(str(image_path))
        image_bgr = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_bgr

    # def __add_to_our_annotation_dict(self, annotation):
    #     image_id = annotation["image_id"]
    #     self.dataset["images"][image_id]["annotations"].append(annotation)

    def get_annotations(self, image_id, return_colors=False):
        # if image_id not in self.annotations_by_image_id:
        if image_id < 0 or image_id >= len(self.dataset["images"]):
            if return_colors:
                return [], []
            return []
        image_name = self.image_paths[image_id]
        annotations = self.dataset["images"][image_name]["annotations"]
        cats = [a["category_id"] for a in annotations]
        colors = [self.category_colors[c] for c in cats]
        if return_colors:
            return annotations, colors
        return annotations

    def delete_annotations(self, image_id, annotation_id):
        image_name = self.image_paths[image_id]
        self.dataset["images"][image_name]["annotations"].pop(annotation_id)

    def add_annotation(self, image_id, category_id, mask):
        if mask is None:
            return
        annotation = parse_mask_to_coco(image_id, self.global_annotation_id, mask, category_id)
        # self.__add_to_our_annotation_dict(annotation)
        image_name = self.image_paths[image_id]
        self.dataset["images"][image_name]["annotations"].append(annotation)
        self.global_annotation_id += 1

    def save_annotation(self):
        with open(self.dataset_json_path, "w") as f:
            json.dump(self.dataset, f, indent=4)
