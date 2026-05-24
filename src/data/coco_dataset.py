"""COCO Detection dataset wrapper for RT-DETR.

Returns images as float tensors [3, H, W] normalized to ImageNet statistics,
and targets as dicts with:
  'labels': LongTensor  [M]     — zero-based COCO category ids (0..79)
  'boxes':  FloatTensor [M, 4]  — (cx, cy, w, h) normalized to [0, 1]
  'image_id': int               — COCO image id (for evaluation)
  'orig_size': (H, W)           — original image dimensions
"""

import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np

# Allow loading partially-downloaded / truncated JPEG files
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from pycocotools.coco import COCO
except ImportError:
    raise ImportError(
        "pycocotools is required. Install with: pip install pycocotools"
    )


# Map from 91-class COCO IDs to 80-class contiguous indices
# (some category IDs are not used in COCO 2017)
COCO91_TO_80: dict[int, int] = {}
_COCO_CATEGORIES_80 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
for _idx, _cat_id in enumerate(_COCO_CATEGORIES_80):
    COCO91_TO_80[_cat_id] = _idx


class COCODetection(Dataset):
    """COCO 2017 detection dataset.

    Args:
        img_folder:  Path to directory containing JPEG images.
        ann_file:    Path to COCO JSON annotation file.
        transforms:  Optional callable applied to (PIL.Image, target) pairs.
                     Should return (tensor_image, transformed_target).
        remove_no_annotations: Skip images with zero annotated objects.
    """

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        remove_no_annotations: bool = True,
    ):
        self.img_folder = Path(img_folder)
        self.coco = COCO(ann_file)
        self.transforms = transforms

        # Get all image IDs
        self.ids = list(sorted(self.coco.imgs.keys()))

        if remove_no_annotations:
            # Keep only images that have at least one valid annotation
            valid_ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                anns = self.coco.loadAnns(ann_ids)
                if any(
                    ann["bbox"][2] > 1 and ann["bbox"][3] > 1
                    for ann in anns
                    if ann.get("category_id") in COCO91_TO_80
                ):
                    valid_ids.append(img_id)
            self.ids = valid_ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        """Load image and annotations for a given index.

        Returns:
            (image_tensor, target_dict) where:
              image_tensor: [3, H, W] float32 tensor (after transforms).
              target_dict:  {
                  'labels': LongTensor [M],
                  'boxes':  FloatTensor [M, 4] — (cx,cy,w,h) in [0,1],
                  'image_id': int,
                  'orig_size': Tuple[int, int],
              }
        """
        img_id = self.ids[index]
        img_info = self.coco.imgs[img_id]

        # Load image
        img_path = self.img_folder / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # PIL returns (W, H)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id not in COCO91_TO_80:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            # Convert xywh (pixel) -> cxcywh (normalized)
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(COCO91_TO_80[cat_id])

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": img_id,
            "orig_size": (orig_h, orig_w),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict]],
) -> tuple[torch.Tensor, list[dict]]:
    """Custom collate function for variable-size detection targets.

    Stacks images into a batch tensor and returns targets as a list of dicts
    (since each image may have a different number of objects).

    Args:
        batch: List of (image_tensor, target_dict) from COCODetection.

    Returns:
        (images, targets) where images is [B, 3, H, W] and targets is
        a list of B dicts.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)
