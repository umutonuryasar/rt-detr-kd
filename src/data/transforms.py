"""Data augmentation and preprocessing transforms for RT-DETR.

Each transform is a callable that takes (PIL.Image, target_dict) and returns
(transformed_PIL_or_tensor, transformed_target_dict).

The target dict must contain:
  'boxes':  FloatTensor [M, 4] — (cx, cy, w, h) normalized to [0, 1].
  'labels': LongTensor  [M].

Other keys (e.g., 'image_id', 'orig_size') are passed through unchanged.

Pipeline:
  Training: Resize -> RandomHorizontalFlip -> ColorJitter ->
            ToTensor -> Normalize -> (optional Mosaic as a separate wrapper)
  Validation: Resize -> ToTensor -> Normalize
"""

import random
import math
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import numpy as np
from typing import Callable, Optional


# ImageNet statistics
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, img: Image.Image, target: dict) -> tuple:
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlip:
    """Flip image and bounding boxes horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, target: dict) -> tuple:
        if random.random() < self.p:
            img = TF.hflip(img)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                # (cx, cy, w, h) -> flip cx: new_cx = 1 - cx
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]
                target["boxes"] = boxes
        return img, target


class Resize:
    """Resize image to (size, size) and scale bounding boxes accordingly.

    Bounding boxes in normalized (cx, cy, w, h) form do not need rescaling
    because they are already in [0, 1] coordinates relative to image size.
    """

    def __init__(self, size: int = 640):
        self.size = size

    def __call__(self, img: Image.Image, target: dict) -> tuple:
        img = TF.resize(img, [self.size, self.size])
        return img, target


class RandomResize:
    """Randomly resize to one of the given sizes (multi-scale training)."""

    def __init__(self, sizes: list[int]):
        self.sizes = sizes

    def __call__(self, img: Image.Image, target: dict) -> tuple:
        size = random.choice(self.sizes)
        img = TF.resize(img, [size, size])
        return img, target


class ColorJitter:
    """Random color jitter (brightness, contrast, saturation, hue)."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        p: float = 0.8,
    ):
        self.p = p
        # (no dummy reference needed)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Image.Image, target: dict) -> tuple:
        if random.random() < self.p:
            # Apply each augmentation independently with random magnitude
            if self.brightness > 0:
                factor = random.uniform(
                    max(0, 1 - self.brightness), 1 + self.brightness
                )
                img = TF.adjust_brightness(img, factor)
            if self.contrast > 0:
                factor = random.uniform(
                    max(0, 1 - self.contrast), 1 + self.contrast
                )
                img = TF.adjust_contrast(img, factor)
            if self.saturation > 0:
                factor = random.uniform(
                    max(0, 1 - self.saturation), 1 + self.saturation
                )
                img = TF.adjust_saturation(img, factor)
            if self.hue > 0:
                factor = random.uniform(-self.hue, self.hue)
                img = TF.adjust_hue(img, factor)
        return img, target


class ToTensor:
    """Convert PIL image to float tensor [3, H, W] in [0, 1]."""

    def __call__(self, img: Image.Image, target: dict) -> tuple:
        img = TF.to_tensor(img)
        return img, target


class Normalize:
    """Normalize tensor image with given mean and std."""

    def __init__(
        self,
        mean: list[float] = _IMAGENET_MEAN,
        std: list[float] = _IMAGENET_STD,
    ):
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor, target: dict) -> tuple:
        img = TF.normalize(img, self.mean, self.std)
        return img, target


class RandomErasing:
    """Random erasing augmentation for regularization (Zhong et al., 2020).

    Operates on the tensor image after ToTensor and Normalize.
    Erases a random rectangle with the mean pixel value (effectively 0 after
    normalization).
    """

    def __init__(self, p: float = 0.25, scale: tuple = (0.02, 0.1)):
        self.p = p
        self.scale = scale

    def __call__(self, img: torch.Tensor, target: dict) -> tuple:
        if random.random() < self.p:
            _, H, W = img.shape
            area = H * W
            erase_area = random.uniform(*self.scale) * area
            aspect = random.uniform(0.3, 3.3)
            eh = int(math.sqrt(erase_area / aspect))
            ew = int(math.sqrt(erase_area * aspect))
            eh, ew = min(eh, H), min(ew, W)
            y0 = random.randint(0, H - eh)
            x0 = random.randint(0, W - ew)
            img = img.clone()
            img[:, y0:y0 + eh, x0:x0 + ew] = 0.0
        return img, target


# ---------------------------------------------------------------------------
# Mosaic augmentation
# ---------------------------------------------------------------------------

class MosaicWrapper:
    """Mosaic augmentation: combines 4 images into a 2×2 grid.

    This wraps a base dataset and applies mosaic with probability `p`.
    When mosaic is not applied, the standard transform pipeline is used.

    Note: Because mosaic requires access to 3 additional random images, it
    must be applied at the dataset level rather than as a simple transform.

    Args:
        dataset: A COCODetection dataset (or any __getitem__-compatible dataset).
        base_transform: Transform applied to individual images before mosaic.
        img_size: Output mosaic size (the 2×2 grid fills img_size × img_size).
        p: Probability of applying mosaic for any given sample.
    """

    def __init__(self, dataset, base_transform=None, img_size: int = 640, p: float = 0.5):
        self.dataset = dataset
        self.base_transform = base_transform
        self.img_size = img_size
        self.p = p

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        if random.random() > self.p:
            # Standard path
            img, target = self.dataset[index]
            if self.base_transform is not None:
                img, target = self.base_transform(img, target)
            return img, target

        # Mosaic path: pick 4 images
        indices = [index] + random.sample(range(len(self.dataset)), 3)
        imgs, targets = [], []
        for i in indices:
            im, tgt = self.dataset[i]
            imgs.append(im)
            targets.append(tgt)

        return self._make_mosaic(imgs, targets)

    def _make_mosaic(
        self, imgs: list, targets: list
    ) -> tuple[torch.Tensor, dict]:
        """Combine 4 images into a single mosaic."""
        s = self.img_size
        half = s // 2

        # Place images in the four quadrants
        # Quadrant layout:  [0|1]
        #                   [2|3]
        canvas = Image.new("RGB", (s, s))
        positions = [
            (0, 0, half, half),
            (half, 0, s, half),
            (0, half, half, s),
            (half, half, s, s),
        ]

        all_boxes = []
        all_labels = []

        for k, (img, target) in enumerate(zip(imgs, targets)):
            x0, y0, x1, y1 = positions[k]
            w_cell, h_cell = x1 - x0, y1 - y0
            # Resize image to cell size
            img_resized = img.resize((w_cell, h_cell)) if isinstance(img, Image.Image) \
                else Image.fromarray(
                    (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                ).resize((w_cell, h_cell))
            canvas.paste(img_resized, (x0, y0))

            boxes = target["boxes"]  # [M, 4] cxcywh normalized
            labels = target["labels"]

            if boxes.numel() > 0:
                # Rescale boxes from [0,1] in cell space to [0,1] in full canvas
                new_cx = (x0 + boxes[:, 0] * w_cell) / s
                new_cy = (y0 + boxes[:, 1] * h_cell) / s
                new_w = boxes[:, 2] * w_cell / s
                new_h = boxes[:, 3] * h_cell / s

                # Clip to canvas
                x1_b = (new_cx + new_w / 2).clamp(0, 1)
                x0_b = (new_cx - new_w / 2).clamp(0, 1)
                y1_b = (new_cy + new_h / 2).clamp(0, 1)
                y0_b = (new_cy - new_h / 2).clamp(0, 1)
                new_w = x1_b - x0_b
                new_h = y1_b - y0_b
                new_cx = (x0_b + x1_b) / 2
                new_cy = (y0_b + y1_b) / 2

                # Filter out degenerate boxes
                valid = (new_w > 1e-4) & (new_h > 1e-4)
                if valid.any():
                    new_boxes = torch.stack(
                        [new_cx[valid], new_cy[valid], new_w[valid], new_h[valid]], dim=1
                    )
                    all_boxes.append(new_boxes)
                    all_labels.append(labels[valid])

        if all_boxes:
            final_boxes = torch.cat(all_boxes, dim=0)
            final_labels = torch.cat(all_labels, dim=0)
        else:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_labels = torch.zeros((0,), dtype=torch.long)

        # Convert canvas to tensor and normalize
        img_tensor = TF.to_tensor(canvas)
        img_tensor = TF.normalize(img_tensor, _IMAGENET_MEAN, _IMAGENET_STD)

        mosaic_target = {
            "boxes": final_boxes,
            "labels": final_labels,
            "image_id": targets[0].get("image_id", -1),
            "orig_size": (s, s),
        }
        return img_tensor, mosaic_target


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_transforms(train: bool = True, img_size: int = 640) -> Compose:
    """Build the standard transform pipeline.

    Args:
        train: If True, include training augmentations.
        img_size: Target image size (square).

    Returns:
        A Compose transform callable.
    """
    if train:
        return Compose(
            [
                Resize(img_size),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
                ToTensor(),
                Normalize(),
                RandomErasing(p=0.25),
            ]
        )
    else:
        return Compose(
            [
                Resize(img_size),
                ToTensor(),
                Normalize(),
            ]
        )
