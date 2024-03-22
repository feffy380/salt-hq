from functools import lru_cache
from pathlib import Path
from threading import Thread

import numpy as np
from segment_anything_hq import SamPredictor

from salt.dataset_explorer import DatasetExplorer
from salt.display_utils import DisplayUtils


class CurrentCapturedInputs:
    def __init__(self):
        self.reset_inputs()

    def reset_inputs(self):
        self.input_points = None
        self.input_labels = None
        self.input_box = None
        self.low_res_logits = None
        self.curr_mask = None

    def set_mask(self, mask):
        self.curr_mask = mask

    def add_input_click(self, input_point, input_label):
        if self.input_points is None:
            self.input_points = np.array([input_point])
            self.input_labels = np.array([input_label])
        else:
            self.input_points = np.vstack([self.input_points, np.array([input_point])])
            self.input_labels = np.append(self.input_labels, input_label)

    def set_input_box(self, bbox):
        if not isinstance(bbox, np.ndarray):
            bbox = np.array(bbox)
        self.input_box = bbox

    def set_low_res_logits(self, low_res_logits):
        self.low_res_logits = low_res_logits


class Editor:
    def __init__(
        self, sam, dataset_path, categories=None, coco_json_path=None
    ):
        self.dataset_path = Path(dataset_path)
        if categories is None and coco_json_path is None:
            raise ValueError("categories must be provided if coco_json_path is None")
        if coco_json_path is None:
            coco_json_path = self.dataset_path / "annotations.json"
        self.coco_json_path = Path(coco_json_path)
        self.dataset_explorer = DatasetExplorer(
            self.dataset_path, categories=categories, coco_json_path=self.coco_json_path
        )
        self.curr_inputs = CurrentCapturedInputs()
        self.categories, self.category_colors = self.dataset_explorer.get_categories(
            get_colors=True
        )
        self.image_id = 0
        self.category_id = 0
        self.show_other_anns = True
        self.sam = sam
        self.predictor = None
        self.du = DisplayUtils()
        self.update_image()

    def list_annotations(self):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        return anns, colors

    def delete_annotations(self, annotation_id):
        self.dataset_explorer.delete_annotations(self.image_id, annotation_id)

    def __draw_known_annotations(self, selected_annotations=[]):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        if selected_annotations:
            anns, colors = zip(*((ann, color) for ann, color in zip(anns, colors) if ann["id"] in selected_annotations))
        # Use this to list the annotations
        self.display = self.du.draw_annotations(self.display, anns, colors)

    def __draw(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        if self.curr_inputs.curr_mask is not None:
            if self.curr_inputs.input_points is not None:
                self.display = self.du.draw_points(
                    self.display, self.curr_inputs.input_points, self.curr_inputs.input_labels
                )
            self.display = self.du.overlay_mask_on_image(
                self.display, self.curr_inputs.curr_mask
            )
        if self.show_other_anns or selected_annotations:
            self.__draw_known_annotations(selected_annotations)

    def update_overlay(self, selected_annotations=[]):
        masks, _, low_res_logits = self.predictor.predict(
            point_coords=self.curr_inputs.input_points,
            point_labels=self.curr_inputs.input_labels,
            box=self.curr_inputs.input_box,
            mask_input=self.curr_inputs.low_res_logits,
            multimask_output=False,
        )
        self.curr_inputs.set_mask(masks[0, :, :])
        self.curr_inputs.set_low_res_logits(low_res_logits)
        self.__draw(selected_annotations)

    def add_click(self, new_pt, new_label, selected_annotations=[]):
        self.curr_inputs.add_input_click(new_pt, new_label)
        self.update_overlay(selected_annotations)

    def remove_click(self, new_pt):
        print("ran remove click")

    def set_bbox(self, bbox, selected_annotations=[]):
        self.curr_inputs.set_input_box(bbox)
        self.update_overlay(selected_annotations)

    def reset(self, hard=True, selected_annotations=[]):
        self.curr_inputs.reset_inputs()
        self.__draw(selected_annotations)

    def toggle(self, selected_annotations=[]):
        self.show_other_anns = not self.show_other_anns
        self.__draw(selected_annotations)

    def step_up_transparency(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        self.du.increase_transparency()
        self.__draw(selected_annotations)

    def step_down_transparency(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        self.du.decrease_transparency()
        self.__draw(selected_annotations)

    def draw_selected_annotations(self, selected_annotations=[]):
        self.__draw(selected_annotations)

    def save_ann(self):
        self.dataset_explorer.add_annotation(
            self.image_id, self.category_id, self.curr_inputs.curr_mask
        )

    def save(self):
        self.dataset_explorer.save_annotation()

    # features are 84MB. don't cache too many
    @lru_cache(maxsize=10)
    def get_cached_image_data(self, image_id):
        if image_id < 0 or image_id >= self.dataset_explorer.get_num_images():
            return None, None, None
        image, image_bgr = self.dataset_explorer.get_image_data(image_id)
        predictor = SamPredictor(self.sam)
        predictor.set_image(image)
        return image, image_bgr, predictor

    def update_image(self):
        self.image, self.image_bgr, self.predictor = self.get_cached_image_data(self.image_id)
        # prefetch 5 images centered around current
        for i in range(self.image_id - 2, self.image_id + 3):
            Thread(
                target=self.get_cached_image_data,
                args=(i,)
            ).start()

        self.display = self.image_bgr.copy()
        self.reset()

    def next_image(self):
        if self.image_id == self.dataset_explorer.get_num_images() - 1:
            return
        self.image_id += 1
        self.update_image()

    def prev_image(self):
        if self.image_id == 0:
            return
        self.image_id -= 1
        self.update_image()

    def fast_forward(self):
        i = 0
        while i < self.dataset_explorer.get_num_images():
            if not self.dataset_explorer.get_annotations(i):
                self.image_id = max(i - 1, 0)
                break
            i += 1
        self.update_image()

    def next_category(self):
        if self.category_id == len(self.categories) - 1:
            self.category_id = 0
            return
        self.category_id += 1

    def prev_category(self):
        if self.category_id == 0:
            self.category_id = len(self.categories) - 1
            return
        self.category_id -= 1

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def select_category(self, category_name):
        category_id = self.categories.index(category_name)
        self.category_id = category_id
