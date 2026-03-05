import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
import json
import argparse
import warnings
from tqdm import tqdm
import sys

# Get the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add src/ to path
sys.path.append(os.path.join(project_root, "src"))

from data.transforms import CocoToFasterRCNN
from data.dataset import SAR_ATR_Dataset
from models.model import get_model

from torchmetrics.detection import MeanAveragePrecision


def collate_fn(batch):

    return tuple(zip(*batch))

def prepare_for_json(dict_results):

    clean_dict = {}
    for k, v in dict_results.items():
        if isinstance(v, torch.Tensor):
            clean_dict[k] = v.detach().cpu().tolist()
        else:
            clean_dict[k] = v
    return clean_dict

def get_class_names(ann_file):
    """Récupère les noms des classes depuis le fichier COCO"""
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Créer un dictionnaire {id: name}
    class_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    return class_names

def validation(num_classes, proportion):

    config_path = project_root / "experiments" / "config" / "config.yaml"
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img_dir = project_root / config["data"]["images"]["test"]["img_dir"]
    ann_file = project_root / config["data"]["annotations"]["test"]["ann_file"]

    # Récupérer les noms des classes
    class_names = get_class_names(ann_file)

    my_transform = CocoToFasterRCNN()
    dataset = SAR_ATR_Dataset(root = str(img_dir), annFile = str(ann_file), transforms=my_transform, subset_ratio=proportion)

    model_path = "../experiments/models/faster_rcnn.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = get_model(num_classes + 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    #model.roi_heads.nms_thresh = 0.3
    model.to(device)
    model.eval() 

    validation_loader = DataLoader(dataset,
                                    batch_size=config["training"]["batch_size"],
                                    shuffle=False,
                                    collate_fn=collate_fn,
                                    num_workers=2,
                                    pin_memory=True
                                )

    metric = MeanAveragePrecision(box_format='xyxy',
                                  class_metrics =True,
                                  )

    results_detailed = []
    with torch.no_grad():
        for images, targets in tqdm(validation_loader) :
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            preds_metrics = []
            for pred in predictions:  
                preds_metrics.append({
                    'boxes': pred['boxes'],
                    'labels': pred['labels'],
                    'scores': pred['scores']
                })
            
            metric.update(preds_metrics, targets)


            for target, pred in zip(targets, preds_metrics):
                results_detailed.append({
                    "ground_truth": {
                        "boxes": target["boxes"].tolist(),
                        "labels": target["labels"].tolist()
                    },
                    "predictions": {
                        "boxes": pred["boxes"].tolist(),
                        "labels": pred["labels"].tolist(),
                        "scores": pred["scores"].tolist()
                    }
                })

    final_results = metric.compute()
    print(final_results)

    json_results = {
        "global_metrics" : prepare_for_json(final_results),
        "class_names": class_names,
        "detailed_results" : results_detailed
    }

    os.makedirs("../experiments/outputs", exist_ok=True)
    with open("../experiments/outputs/test_results.json", "w", encoding="utf-8") as file:
        json.dump(json_results, file, indent=2, ensure_ascii=False)
    print("Results saved")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--proportion', type=float, default=1.0, help='proportion of the original dataset')
    args = parser.parse_args()
    validation(args.num_classes, args.proportion)
