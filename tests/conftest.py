"""Shared pytest fixtures and helpers.

All tests are designed to run on CPU with tiny tensors so CI can execute in
seconds without a GPU. Anything that requires CUDA is skipped automatically.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# Add repo root to import path so `import src.*` works without packaging.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _deterministic():
    """Each test starts from a fixed seed."""
    torch.manual_seed(0)


@pytest.fixture
def tiny_images() -> torch.Tensor:
    """2 normalized RGB images at 128x128 — enough to exercise the pipeline
    without paying for real 640x640 forward passes."""
    return torch.randn(2, 3, 128, 128)


@pytest.fixture
def tiny_targets():
    """Minimal valid COCO-format targets for a batch of 2 images."""
    return [
        {
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "boxes":  torch.tensor([[0.5, 0.5, 0.3, 0.3],
                                    [0.2, 0.2, 0.1, 0.1]], dtype=torch.float32),
            "image_id": 1,
            "orig_size": (128, 128),
        },
        {
            "labels": torch.tensor([2], dtype=torch.long),
            "boxes":  torch.tensor([[0.7, 0.7, 0.2, 0.2]], dtype=torch.float32),
            "image_id": 2,
            "orig_size": (128, 128),
        },
    ]


def skip_if_no_cuda():
    """Helper for tests that genuinely need CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
