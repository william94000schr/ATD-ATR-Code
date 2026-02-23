import os
import json
import random
import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from transforms import CocoToFasterRCNN
from dataset import SAR_ATR_Dataset
from model import get_model
from visualization import save_prediction, save_gradcam
from gradcam import FasterRCNNGradCAM


OUTPUT_DIR = "../outputs/predictions"


def load_config():
    with open("../config/config.yaml", 'r') as f:
        return yaml.safe_load(f)


def load_class_names():
    path = "../config/classes.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_model(device):
    checkpoint = torch.load("../models/faster_rcnn.pt", map_location=device)
    # Le predictor sauvegardé contient la bonne shape
    num_classes = checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0]
    
    model = get_model(num_classes)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Model loaded — {num_classes} classes (including background)")
    return model


def run_on_image(image_path, model, device, threshold, explainability, class_names):
    to_tensor = transforms.ToTensor()
    orig_image = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(orig_image)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if explainability:
        gradcam = FasterRCNNGradCAM(model)
        preds, heatmaps = gradcam.compute(image_tensor, device, threshold)
        gradcam.remove_hooks()
        if preds is None:
            print("No detection above threshold.")
            return
        save_path = f"{OUTPUT_DIR}/gradcam_single.png"
        save_gradcam(orig_image, preds, heatmaps, save_path, class_names=class_names)
        print(f"Saved: {save_path}")
    else:
        with torch.no_grad():
            preds = model([image_tensor.to(device)])[0]
        keep = preds['scores'] > threshold
        filtered = {k: v[keep] for k, v in preds.items()}
        save_path = f"{OUTPUT_DIR}/pred_single.png"
        save_prediction(orig_image, filtered, save_path, class_names=class_names)
        print(f"Saved: {save_path}")


def run_on_dataset(num_images, model, device, threshold, explainability, class_names, config):
    project_root = Path(__file__).parent.parent
    img_dir  = project_root / config["data"]["images"]["test"]["img_dir"]
    ann_file = project_root / config["data"]["annotations"]["test"]["ann_file"]

    dataset = SAR_ATR_Dataset(
        root=str(img_dir), annFile=str(ann_file),
        transforms=CocoToFasterRCNN()
    )

    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gradcam = FasterRCNNGradCAM(model) if explainability else None

    for idx in indices:
        image_tensor, target = dataset[idx]
        orig_image = dataset._load_image(dataset.ids[idx])

        ground_truth = {'boxes': target['boxes'], 'labels': target['labels']}

        if explainability:
            preds, heatmaps = gradcam.compute(image_tensor, device, threshold)
            if preds is None:
                print(f"[{idx}] No detection above threshold.")
                continue
            save_path = f"{OUTPUT_DIR}/gradcam_{idx}.png"
            save_gradcam(orig_image, preds, heatmaps, save_path, ground_truth=ground_truth, class_names=class_names)
        else:
            with torch.no_grad():
                preds = model([image_tensor.to(device)])[0]
            keep = preds['scores'] > threshold
            filtered = {k: v[keep] for k, v in preds.items()}
            save_path = f"{OUTPUT_DIR}/pred_{idx}.png"
            save_prediction(orig_image, filtered, save_path, ground_truth=ground_truth, class_names=class_names)

        print(f"[{idx}] Saved: {save_path}")

    if gradcam:
        gradcam.remove_hooks()


def main():
    parser = argparse.ArgumentParser(description="Faster RCNN inference")
    parser.add_argument('--image_path',     type=str,   default=None,  help='Path to a single image')
    parser.add_argument('--num_images',     type=int,   default=1,     help='Number of images from test dataset')
    parser.add_argument('--threshold',      type=float, default=0.5,   help='Detection score threshold')
    parser.add_argument('--explainability', action='store_true',       help='Enable GradCAM visualization')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    config      = load_config()
    class_names = load_class_names()
    model       = load_model(device)

    if args.image_path:
        run_on_image(args.image_path, model, device, args.threshold, args.explainability, class_names)
    else:
        run_on_dataset(args.num_images, model, device, args.threshold, args.explainability, class_names, config)


if __name__ == "__main__":
    main()