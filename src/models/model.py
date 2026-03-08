from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)  # type: ignore

# from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights  # noqa: E501
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # type: ignore


def get_model(num_classes):
    """Build a Faster R-CNN ResNet-50 FPN v2 model with a custom classification head.

    Args:
        num_classes: Total number of classes including background.

    Returns:
        A Faster R-CNN model with the box predictor replaced to match ``num_classes``.
    """
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=5,
        min_size=128,
        max_size=128,
    )
    # model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,  # noqa: E501
    # trainable_backbone_layers =6,
    # min_size = 128,
    # max_size = 128)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
