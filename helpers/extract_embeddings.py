# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache-2.0 license found in the LICENSE file in the root directory of segment_anything repository and source tree.
# Adapted from onnx_model_example.ipynb in the segment_anything repository.
# Please see the original notebook for more details and other examples and additional usage.
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything_hq import SamPredictor, sam_model_registry
from tqdm import tqdm


# TODO: refactor to allow on-demand mask extraction
def main(checkpoint_path, model_type, device, images_folder, embeddings_folder):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_paths = [
        path
        for path in images_folder.iterdir()
        if path.suffix.lower() in [".jpg", ".png"]
    ]
    for image_path in tqdm(image_paths):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        image_embedding = predictor.get_image_embedding().cpu().numpy()
        interm_embeddings = torch.stack(predictor.interm_features).cpu().numpy()

        out_path = embeddings_folder / image_path.with_suffix(".npy").name
        interm_out_path = embeddings_folder / (image_path.stem + "_interm.npy")
        np.save(out_path, image_embedding)
        np.save(interm_out_path, interm_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="./sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, default="./example_dataset")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    device = args.device
    dataset_path = Path(args.dataset_path)

    images_folder = dataset_path / "images"
    embeddings_folder = dataset_path / "embeddings"
    embeddings_folder.mkdir(exist_ok=True)

    main(checkpoint_path, model_type, device, images_folder, embeddings_folder)
