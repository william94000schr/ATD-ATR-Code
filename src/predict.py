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


def collate_fn(batch):

    return tuple(zip(*batch))

def predict(num_classes, num_images, image_path, threshold, proportion):

    if image_path:
        # Charger l'image unique
        orig_image = Image.open(image_path).convert("RGB")
        image_tensor = my_transform(orig_image, None)[0]  # Juste l'image, pas de target
        
        with torch.no_grad():
            prediction = model([image_tensor.to(device)])[0]
            draw = ImageDraw.Draw(orig_image)
            
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score > threshold: 
                    b = box.cpu().numpy()
                    draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="red", width=3)
                    text2 = f"{label.item()} : {score:.2f} %"
                    bbox2 = draw.textbbox((b[2], b[1] - 15), text2)
                    draw.rectangle(bbox2, fill="red")
                    draw.text((b[2], b[1] - 15), text2, fill="white")
        
            orig_image.save(f"../outputs/predictions/single_pred.png")
    else:

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
                orig_image = dataset._load_image(dataset.ids[idx])

                #boxes
                prediction = model([image_tensor.to(device)])[0]
                draw = ImageDraw.Draw(orig_image)


                for box, label in zip(target['boxes'], target['labels']):
                    b = box.numpy()
                    draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="green", width=2)
                    text = str(label.item())
                    bbox = draw.textbbox((b[0], b[1] - 15), text)
                    draw.rectangle(bbox, fill="green")
                    draw.text((b[0], b[1] - 15), text, fill="white")

        
                for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                    if score > threshold: 
                        b = box.cpu().numpy()
                        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="red", width=3)
                        text2 = f"{label.item()} : {score:.2f} %"
                        bbox2 = draw.textbbox((b[2], b[1] - 15), text2)
                        draw.rectangle(bbox2, fill="red")
                        draw.text((b[2], b[1] - 15), text2, fill="white")


                print(f"image {idx}")
                print("\nVérité Terrain:")
                for box, label in zip(target['boxes'], target['labels']):
                    print(f"- Classe: {label.item()}\n")

                print("Prédictions du modèle:")
                for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                    if score > threshold: 
                        print(f"- Classe: {label.item()} | Score: {score:.2f}\n")

                orig_image.save(f"../outputs/predictions/pred_{idx}.png")
                print('-'*60)
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--num_images', type=int, default=1, help='number of images we want to visualize')
    parser.add_argument('--image_path', type=str, default=None, help='Path to an image')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of detection')
    parser.add_argument('--proportion', type=float, default=1.0, help='proportion of the original dataset')
    args = parser.parse_args()
    predict(args.num_classes, args.num_images, args.image_path, args.threshold, args.proportion)
