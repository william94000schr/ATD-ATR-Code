import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import get_model
from visualization import save_prediction, save_gradcam
from gradcam import FasterRCNNGradCAM


OUTPUT_DIR = "../outputs/predictions"


def load_class_names():
    path = "../config/classes.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_model(device):
    checkpoint = torch.load("../models/faster_rcnn.pt", map_location=device)
    num_classes = checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0]
    model = get_model(num_classes)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Model loaded — {num_classes} classes (including background)")
    return model


def run_on_image(orig_image, image_tensor, model, device, threshold, explainability=False, ground_truth=None, class_names=None, save_name="pred"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if explainability:
        gradcam = FasterRCNNGradCAM(model)
        preds, heatmaps = gradcam.compute(image_tensor, device, threshold)
        gradcam.remove_hooks()
        if preds is None:
            print("No detection above threshold.")
            return
        save_path = f"{OUTPUT_DIR}/gradcam_{save_name}.png"
        save_gradcam(orig_image, preds, heatmaps, save_path, ground_truth=ground_truth, class_names=class_names)
    else:
        with torch.no_grad():
            preds = model([image_tensor.to(device)])[0]
        keep = preds['scores'] > threshold
        filtered = {k: v[keep] for k, v in preds.items()}
        save_path = f"{OUTPUT_DIR}/{save_name}.png"
        save_prediction(orig_image, filtered, save_path, ground_truth=ground_truth, class_names=class_names)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster RCNN inference")
    parser.add_argument('--image_path',     type=str,   required=True, help='Path to image')
    parser.add_argument('--threshold',      type=float, default=0.5,   help='Detection score threshold')
    parser.add_argument('--explainability', action='store_true',       help='Enable GradCAM visualization')
    args = parser.parse_args()

    device      = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    class_names = load_class_names()
    model       = load_model(device)

    orig_image   = Image.open(args.image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(orig_image)

    run_on_image(orig_image, image_tensor, model, device, args.threshold, args.explainability, class_names=class_names)