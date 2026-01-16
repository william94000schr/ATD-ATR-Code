import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Union
from PIL import Image
from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO


class SAR_ATR_Dataset(VisionDataset):

    def __init__(self, root: Union[str, Path], annFile: str, transforms) -> None:
        
        super().__init__(root, transforms= transforms)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> list[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def __len__(self) -> int:
        return len(self.ids)