import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch  # type: ignore
import yaml
from torch.cuda.amp import GradScaler, autocast  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

# Get the project root
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src/ to path
sys.path.append(os.path.join(project_root, "src"))
from tqdm import tqdm  # type: ignore

from data.dataset import SAR_ATR_Dataset
from data.transforms import CocoToFasterRCNN
from models.model import get_model

warnings.filterwarnings("ignore", category=FutureWarning)


def collate_fn(batch):
    return tuple(zip(*batch))


def update_learning_rate(optimizer, epoch, lr_schedule):
    for stage in lr_schedule:
        if stage["epochs"][0] <= epoch < stage["epochs"][1]:
            new_lr = stage["lr"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            return new_lr
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(model, train_loader, optimizer, device, scaler, use_amp, epoch):
    """Entraîne le modèle sur une epoch"""
    model.train()

    result = {
        "loss_classifier": 0,
        "loss_box_reg": 0,
        "loss_objectness": 0,
        "loss_rpn_box_reg": 0,
        "loss_total": 0,
    }
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                loss_dict = model(images, targets)
                losses_epoch = sum(loss for loss in loss_dict.values())
            scaler.scale(losses_epoch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses_epoch = sum(loss for loss in loss_dict.values())
            losses_epoch.backward()
            optimizer.step()

        result["loss_classifier"] += loss_dict["loss_classifier"].item()
        result["loss_box_reg"] += loss_dict["loss_box_reg"].item()
        result["loss_objectness"] += loss_dict["loss_objectness"].item()
        result["loss_rpn_box_reg"] += loss_dict["loss_rpn_box_reg"].item()
        result["loss_total"] += losses_epoch.item()

        pbar.set_postfix(
            {
                "total": f"{losses_epoch.item():.3f}",
                "cls": f"{loss_dict['loss_classifier'].item():.3f}",
                "box": f"{loss_dict['loss_box_reg'].item():.3f}",
                "obj": f"{loss_dict['loss_objectness'].item():.3f}",
            }
        )

        num_batches += 1

    # Moyenne des losses
    for key in result:
        result[key] /= num_batches

    return result


def main(num_classes, num_epochs, proportion):

    # Configuration
    project_root = Path(__file__).parent.parent
    config_path = project_root / "experiments" / "config" / "config.yaml"
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Chargement des données
    img_dir = project_root / config["data"]["images"]["train"]["img_dir"]
    ann_file = project_root / config["data"]["annotations"]["train"]["ann_file"]
    my_transform = CocoToFasterRCNN()
    full_dataset = SAR_ATR_Dataset(
        root=str(img_dir), annFile=str(ann_file), transforms=my_transform, subset_ratio=proportion
    )

    # DataLoaders
    train_loader = DataLoader(
        full_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Modèle
    model = get_model(num_classes + 1)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    initial_lr = config["training"]["learning_rate_schedule"][0]["lr"]
    optimizer = torch.optim.SGD(
        params,
        lr=initial_lr,
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Learning rate schedule desde config
    lr_schedule = config["training"]["learning_rate_schedule"]
    print("\nLearning Rate Schedule:")
    for stage in lr_schedule:
        print(f"  Epochs {stage['epochs'][0]}-{stage['epochs'][1] - 1}: lr={stage['lr']}")

    # Mixed precision
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    scaler = GradScaler() if use_amp else None
    torch.backends.cudnn.benchmark = True

    print(f"Using AMP: {use_amp}")

    # Boucle d'entraînement
    results = []
    for epoch in range(num_epochs):
        # Actualizar learning rate según el schedule
        current_lr = update_learning_rate(optimizer, epoch, lr_schedule)

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs} | Learning Rate: {current_lr}")
        print(f"{'=' * 60}")

        # Train
        train_result = train_one_epoch(
            model, train_loader, optimizer, device, scaler, use_amp, epoch
        )

        # Affichage
        print(f"\n--- Epoch {epoch} Results ---")
        print("TRAIN:")
        for key, value in train_result.items():
            print(f"  {key}: {value:.4f}")

        # Stockage des résultats
        results.append({"epoch": epoch, "learning_rate": current_lr, "train": train_result})

        if (epoch + 1) % 2 == 0:
            os.makedirs("../experiments/models", exist_ok=True)
            torch.save(model.state_dict(), f"../experiments/models/checkpoint_epoch{epoch + 1}.pt")

    # Sauvegarde
    os.makedirs("../experiments/outputs", exist_ok=True)
    with open("../experiments/outputs/train_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    os.makedirs("../experiments/models", exist_ok=True)
    torch.save(model.state_dict(), "../experiments/models/faster_rcnn.pt")

    print("Training completed!")
    print("Model saved to: ../models/faster_rcnn.pt")
    print("Results saved to: ../outputs/train_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument(
        "--proportion", type=float, default=1.0, help="proportion of the original dataset"
    )
    args = parser.parse_args()
    main(args.num_classes, args.num_epochs, args.proportion)
