from omni.replicator.core import CocoWriter, AnnotatorRegistry
from pycocotools import mask as mask_utils
import numpy as np
from pathlib import Path


class CocoInstanceSegWriter(CocoWriter):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.annotators.append(AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"semanticTypes": self.semantic_types}))
        self.label_dict = {
            'pallet': {'name': 'pallet', 'id': 1, 'supercategory': 'loads', 'color': (220, 20, 60), 'isthing': 1},
            'kkp': {'name': 'KKP', 'id': 2, 'supercategory': 'loads', 'color': (220, 20, 60), 'isthing': 1},
            'goods': {'name': 'goods', 'id': 3, 'supercategory': 'loads', 'color': (119, 11, 32), 'isthing': 1}
        }
        
    def write(self, data: dict):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        """
        # Check for on_time triggers
        # For each on_time trigger, prefix the output frame number with the trigger counts
        sequence_id = ""
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = f"{call_count}_{sequence_id}_"
        if sequence_id != self._sequence_id:
            self._frame_id = 0
            self._sequence_id = sequence_id

        # Loop through all annotators and render products
        for render_product_name, rp_data_dict in data["renderProducts"].items():

            camera_name = rp_data_dict["camera"][1:].replace("Replicator/", "").replace("/", "-").replace("_", "-")
            rgb_path = self._write_rgb(render_product_name, camera_name, rp_data_dict["rgb"])
            image_id = len(self.coco_annotation_dict["images"])
            image_dict = {
                "file_name": Path(rgb_path).as_posix(),
                "id": image_id,
                "height": int(rp_data_dict["resolution"][1]),
                "width": int(rp_data_dict["resolution"][0]),
                "license": 0,
                "date_captured": self._date.isoformat(),
                "flicker_url": "",
            }
            self.coco_annotation_dict["images"].append(image_dict)
            self._write_instance_annotation_segment(rp_data_dict["instance_segmentation"], image_id)

        self._frame_id += 1
        if self._frame_id % 25 == 0:
            # periodically write the annotation file to avoid data loss
            self._write_coco_annotation_file()


    def _write_instance_annotation_segment(self, annotator_dict, image_id):
        instance_map = annotator_dict["data"]
        id_to_semantics = annotator_dict["idToSemantics"]

        image_annotations = []

        for instance_id in np.unique(instance_map):
            instance_id = int(instance_id)
            if instance_id == 0:
                continue

            mask = (instance_map == instance_id).astype(np.uint8)
            area = int(mask.sum())
            if area == 0:
                continue

            ys, xs = np.where(mask > 0)
            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())

            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

            semantic_dict = id_to_semantics.get(str(instance_id), {})
            category_id = None
            for label in semantic_dict.values():
                if label in self.label_dict:
                    category_id = self.label_dict[label]["id"]
                    self._used_categories.setdefault(label, self.label_dict[label])
                

            if category_id is None:
                continue

            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            annotation_entry = {
                "id": self._num_annotations,
                "image_id": image_id,
                "bbox": bbox,
                "area": float(area),
                "bbox_mode": 1,
                "category_id": int(category_id),
                "iscrowd": 0,
                "segmentation": rle
            }

            image_annotations.append(annotation_entry)
            self._num_annotations += 1

        self.coco_annotation_dict["annotations"] += image_annotations