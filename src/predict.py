import os
from pathlib import Path
import torch
import yaml
import argparse
import random
from transforms import CocoToFasterRCNN
from dataset import SAR_ATR_Dataset
from model import get_model
from PIL import Image, ImageDraw
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


def collate_fn(batch):

    return tuple(zip(*batch))

def predict(num_classes, num_images, threshold, proportion):

    project_root = Path(__file__).parent.parent 
    with open("../config/config.yaml", 'r') as stream:
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
    
    os.makedirs("../outputs/predictions", exist_ok=True)
    indices = random.sample(range(len(dataset)), num_images)
    with torch.no_grad():
        for idx in indices:
            image_tensor, target = dataset[idx]

            # Obligé de passer par une liste dans le modèle
            # -> On récupère donc une liste (contenant un seul élément)            
            prediction = model([image_tensor.to(device)])[0]
            
            #Bounding box pour la vérité terrain
            img_with_GT = draw_bounding_boxes(image_tensor,
                                boxes = target["boxes"],
                                labels = [str(i.item()) for i in target["labels"]], # Liste de String obligatoire
                                colors = "green",
                                width = 3,
                                label_colors = "white",
                                label_background_colors = "green",
                                )
            
            #Bounding box pour la prédiction
            img_with_GT_and_preds = draw_bounding_boxes(img_with_GT,
                                boxes = prediction["boxes"],
                                labels = [f"{label} : {score}" for label,score in zip(prediction["labels"], prediction["scores"])],
                                colors = "red",
                                width = 3,
                                label_colors = "white",
                                label_background_colors = "red",
                                )

            print(idx)
            print("\nVérité Terrain:")
            for box, label in zip(target['boxes'], target['labels']):
                print(f"  Classe: {label.item()}\n")

            print("Prédictions du modèle:")
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score > threshold: 
                    print(f"Classe: {label.item()} | Score: {score:.2f}\n")

            to_pil_image(img_with_GT_and_preds).save(f"../outputs/predictions/pred_{idx}.png")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--num_images', type=int, default=1, help='number of images we want to visualize')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of detection')
    parser.add_argument('--proportion', type=float, default=1.0, help='proportion of the original dataset')
    args = parser.parse_args()
    predict(args.num_classes, args.num_images, args.threshold, args.proportion)
