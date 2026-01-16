import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
import json
import argparse
from transforms import CocoToFasterRCNN
from dataset import SAR_ATR_Dataset
from model import get_model
from tqdm import tqdm

def collate_fn(batch):

    return tuple(zip(*batch))

def train(num_classes, num_epochs, proportion):

    num_classes = num_classes

    project_root = Path(__file__).parent.parent 
    config_path = project_root / "config" / "config.yaml"
    with open("../config/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img_dir = project_root / config["data"]["images"]["train"]["img_dir"]
    ann_file = project_root / config["data"]["annotations"]["train"]["ann_file"]
    my_transform = CocoToFasterRCNN()
    dataset = SAR_ATR_Dataset(root = str(img_dir), annFile = str(ann_file), transforms=my_transform, subset_ratio=proportion)

    train_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = get_model(num_classes + 1)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr= config["training"]["learning_rate"], momentum=config["training"]["momentum"], weight_decay=config["training"]["weight_decay"])

    torch.backends.cudnn.benchmark = True 
    model.train()

    results = []
    for epoch in tqdm(range(num_epochs), desc = "Epochs" ):

        result = {
            "epoch": epoch,
            "loss_classifier": 0,
            "loss_box_reg": 0,
            "loss_objectness": 0,
            "loss_rpn_box_reg": 0,
            "loss_total": 0
        }
        num_batches = 0

        for images, targets in train_loader:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            result["loss_classifier"] += loss_dict["loss_classifier"].item()
            result["loss_box_reg"] += loss_dict["loss_box_reg"].item()
            result["loss_objectness"] += loss_dict["loss_objectness"].item()
            result["loss_rpn_box_reg"] += loss_dict["loss_rpn_box_reg"].item()
            losses_epoch = sum(loss for loss in loss_dict.values())
            result["loss_total"] += losses_epoch.item()
            
            optimizer.zero_grad()
            losses_epoch.backward()
            optimizer.step()

            num_batches += 1

        for key in result:
            if key != "epoch":
                result[key] /= num_batches
        results.append(result)
    
        print(f"\nEpoch {epoch}:")
        for key, value in result.items():
            if key != "epoch":
                print(f"  {key}: {value:.4f}")


    os.makedirs("./outputs", exist_ok=True)
    with open("../outputs/train_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "../models/faster_rcnn.pt")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--proportion', type=float, default=1.0, help='proportion of the original dataset')
    args = parser.parse_args()
    train(args.num_classes, args.num_epochs, args.proportion)