import os.path
import random
from pathlib import Path
from typing import Any

from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset


class SAR_ATR_Dataset(VisionDataset):
    """PyTorch dataset for SAR ATR images with COCO-format annotations."""

    def __init__(self, root: str | Path, annFile: str, transforms, subset_ratio=1.0) -> None:
        """Initialize the dataset.

        Args:
            root: Root directory containing the images.
            annFile: Path to the COCO JSON annotation file.
            transforms: Transform callable applied to (image, target) pairs.
            subset_ratio: Fraction of the dataset to use (default: 1.0).
        """
        super().__init__(root, transforms=transforms)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        if subset_ratio < 1.0:
            n_samples = int(len(self.ids) * subset_ratio)
            random.seed(94)
            self.ids = random.sample(self.ids, n_samples)

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> list[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Return the (image, target) pair at the given index.

        Args:
            index: Integer index into the dataset.

        Returns:
            Tuple of (image, target) after applying transforms.

        Raises:
            ValueError: If ``index`` is not an integer.
        """
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ids)
