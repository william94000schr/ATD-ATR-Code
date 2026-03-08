import torch
import torchvision.transforms as transforms


class CocoToFasterRCNN:
    """Transform COCO annotations and images into Faster R-CNN input format."""

    def __call__(self, images, targets):
        """Convert a PIL image and COCO annotation list to tensors.

        Args:
            images: PIL image.
            targets: List of COCO annotation dicts.

        Returns:
            Tuple of (image_tensor, target_dict) where target_dict contains
            ``boxes`` (FloatTensor[N, 4]) and ``labels`` (Int64Tensor[N]).
        """
        # Concversion de l'image
        image_tensor = transforms.ToTensor()(images)

        # Conversion de l'annotation
        all_boxes = []
        all_labels = []

        for annotations in targets:
            bbox = annotations["bbox"]
            new_box = self.ConvertBboxToBoxes(bbox)
            all_boxes.append(new_box)

            all_labels.append(annotations["category_id"])

        # if it is not an object , we creat an empty tensor
        if len(all_boxes) > 0:
            boxes = torch.as_tensor(all_boxes, dtype=torch.float32)
            labels = torch.as_tensor(all_labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        new_targets = {"boxes": boxes, "labels": labels}

        return image_tensor, new_targets

    def ConvertBboxToBoxes(self, bbox_coord):
        """Convert COCO ``[x, y, w, h]`` format to ``[x1, y1, x2, y2]`` format."""
        x1, y1, w, h = bbox_coord
        return [x1, y1, x1 + w, y1 + h]
