import argparse
import sys
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from segment_anything_hq import sam_model_registry

from PyQt5.QtWidgets import QApplication

from salt.editor import Editor
from salt.interface import ApplicationInterface
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="./models/sam_hq_vit_h.pth")
    parser.add_argument("--model-type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset-path", type=str, default="./dataset")
    parser.add_argument("--categories", type=str)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    categories = None
    if args.categories is not None:
        categories = args.categories.split(",")

    dataset_json_path = dataset_path / "annotations.json"

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    sam.to(device=args.device)

    editor = Editor(
        sam,
        dataset_path,
        categories=categories,
        dataset_json_path=dataset_json_path
    )

    app = QApplication(sys.argv)
    window = ApplicationInterface(app, editor)
    window.show()
    sys.exit(app.exec_())