import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
from transforms import CocoToFasterRCNN
from dataset import SAR_ATR_Dataset
from model import get_model
from tqdm import tqdm

def collate_fn(batch):

    return tuple(zip(*batch))

def train(num_classes, num_epochs):

    num_classes = num_classes

    project_root = Path(__file__).parent.parent 
    config_path = project_root / "config" / "config.yaml"
    with open("../config/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img_dir = project_root / config["data"]["images"]["train"]["img_dir"]
    ann_file = project_root / config["data"]["annotations"]["train"]["ann_file"]
    my_transform = CocoToFasterRCNN()
    dataset = SAR_ATR_Dataset(root = str(img_dir), annFile = str(ann_file), transforms=my_transform)

    train_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes + 1)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr= config["training"]["learning_rate"], momentum=config["training"]["momentum"], weight_decay=config["training"]["weight_decay"])


    model.train()

    for epoch in tqdm(range(num_epochs), desc = "Epochs" ):
        for images, targets in tqdm(train_loader, desc = "Batches", leave = False):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch} terminée. Loss: {losses.item()}")
    
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), "../models/faster_rcnn.pt")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    args = parser.parse_args()
    train(args.num_classes, args.num_epochs)