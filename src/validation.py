import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
import json
import argparse
import warnings
from tqdm import tqdm
from transforms import CocoToFasterRCNN
from dataset import SAR_ATR_Dataset
from model import get_model
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision


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

def validation(num_classes, proportion, score_threshold):

    project_root = Path(__file__).parent.parent 
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img_dir = project_root / config["data"]["images"]["test"]["img_dir"]
    ann_file = project_root / config["data"]["annotations"]["test"]["ann_file"]
    my_transform = CocoToFasterRCNN()
    dataset = SAR_ATR_Dataset(root = str(img_dir), annFile = str(ann_file), transforms=my_transform, subset_ratio=proportion)

    model = get_model(num_classes + 1)
    model.load_state_dict(torch.load("../models/faster_rcnn.pt", map_location=device))
    model.to(device)
    model.eval() 

    validation_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    metric1 = IntersectionOverUnion()
    metric2 = MeanAveragePrecision(box_format='xyxy')

    results_detailed = []
    with torch.no_grad():
        for images, targets in tqdm(validation_loader) :
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)

            preds_filtered = []
            for pred in preds:
                
                mask = pred['scores'] > score_threshold
                preds_filtered.append({
                    'boxes': pred['boxes'][mask],
                    'labels': pred['labels'][mask],
                    'scores': pred['scores'][mask]
                })
            
            metric1.update(preds_filtered, targets)
            metric2.update(preds_filtered, targets)

            for target, pred in zip(targets, preds_filtered):
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

    final_IoU = metric1.compute()
    final_results = metric2.compute()
    print(final_IoU)
    print(final_results)

    json_results = {
        "global_metrics" :{
            "mAP_metrics": prepare_for_json(final_results),
            "IoU_metrics": prepare_for_json(final_IoU)
            },
        "detailed_results" : results_detailed
    }
    os.makedirs("../outputs", exist_ok=True)
    with open("../outputs/test_results.json", "w", encoding="utf-8") as file:
        json.dump(json_results, file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--proportion', type=float, default=1.0, help='proportion of the original dataset')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='threshold for box we eliminate')
    args = parser.parse_args()
    validation(args.num_classes, args.proportion, args.score_threshold)
