"""
GradCAM pour Faster RCNN ResNet50 FPN v2
Usage:
    python gradcam.py --num_classes 10 --image_path path/to/image.jpg --threshold 0.5
    python gradcam.py --num_classes 10 --threshold 0.5 --num_images 5   # sur le dataset de test
    python gradcam.py --num_classes 10 --image_path path/to/image.jpg --target_class 3  # forcer une classe cible
"""

import os
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
import argparse
import random
import numpy as np
from transforms import CocoToFasterRCNN
from dataset import SAR_ATR_Dataset
from model import get_model
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────
#  GradCAM Engine
# ─────────────────────────────────────────────

class FasterRCNNGradCAM:
    """
    GradCAM adapté à Faster RCNN avec backbone ResNet50 FPN.
    
    On cible backbone.body.layer4 : dernière couche conv avant le FPN.
    Pour chaque détection, on backpropage le score de classification
    de la box choisie, ce qui permet de voir quelles zones ont influencé
    la décision pour CETTE détection spécifique.
    """

    def __init__(self, model, target_layer_name="backbone.body.layer4"):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []
        
        # Récupérer la couche cible par son nom
        target_layer = self._get_layer(target_layer_name)
        
        # Hook forward : capture les activations
        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        # Hook backward : capture les gradients
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )
        
        print(f"[GradCAM] Couche cible : {target_layer_name}")

    def _get_layer(self, layer_name):
        layer = self.model
        for part in layer_name.split("."):
            layer = getattr(layer, part)
        return layer

    def _save_activation(self, module, input, output):
        # output shape : (B, C, H, W)
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] shape : (B, C, H, W)
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def compute(self, image_tensor, device, threshold=0.5, target_class=None):
        """
        Fait le forward pass et calcule les heatmaps GradCAM
        pour toutes les détections au-dessus du seuil.

        Args:
            image_tensor : tensor (C, H, W) non batché
            device       : torch device
            threshold    : seuil de score pour filtrer les détections
            target_class : si spécifié, ne garder que les détections de cette classe

        Returns:
            predictions  : dict avec boxes, labels, scores (filtrés)
            heatmaps     : liste de np.array (H, W) normalisés [0,1], un par détection
        """
        self.model.train()  # IMPORTANT : eval() désactive les gradients dans certaines couches
                            # mais on reste en mode inférence grâce à torch.no_grad() absent

        image = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)

        # Forward pass — PAS de torch.no_grad() pour garder le graphe de calcul
        predictions = self.model(image)[0]

        # Filtrer par seuil et classe cible
        keep = predictions['scores'] > threshold
        if target_class is not None:
            keep = keep & (predictions['labels'] == target_class)

        boxes  = predictions['boxes'][keep]
        labels = predictions['labels'][keep]
        scores = predictions['scores'][keep]

        if len(scores) == 0:
            print("[GradCAM] Aucune détection au-dessus du seuil.")
            return None, []

        heatmaps = []

        for i in range(len(scores)):
            # Remettre les gradients à zéro avant chaque backward
            self.model.zero_grad()
            self.gradients = None

            # Backpropage sur le score de la i-ème détection
            # C'est ce score qui "représente" la confiance du modèle pour cette box
            scores[i].backward(retain_graph=(i < len(scores) - 1))

            if self.gradients is None:
                print(f"[GradCAM] Pas de gradient pour la détection {i}, skip.")
                heatmaps.append(None)
                continue

            # GAP sur les gradients → poids par canal  (C,)
            weights = self.gradients.mean(dim=(2, 3))[0]  # (C,)

            # Combinaison linéaire pondérée des feature maps  (H, W)
            cam = (weights[:, None, None] * self.activations[0]).sum(dim=0)  # (H, W)

            # ReLU : on ne garde que les activations positives
            cam = F.relu(cam)

            # Normalisation [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)

            heatmaps.append(cam.cpu().numpy())

        filtered_preds = {
            'boxes':  boxes.detach(),
            'labels': labels.detach(),
            'scores': scores.detach()
        }

        return filtered_preds, heatmaps


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def overlay_heatmap(pil_image, cam, alpha=0.45, colormap='jet'):
    """
    Superpose une heatmap GradCAM sur une image PIL.
    
    Returns:
        PIL.Image avec la heatmap en overlay
    """
    orig_w, orig_h = pil_image.size

    # Redimensionner la CAM à la taille de l'image originale
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0

    # Appliquer le colormap
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(cam_resized)  # (H, W, 4) en [0,1]
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil  = Image.fromarray(heatmap_rgb)

    # Blend
    orig_np = np.array(pil_image.convert("RGB"))
    blended = (orig_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def draw_boxes_on_image(pil_image, preds, class_names=None):
    """Dessine les boîtes de détection sur l'image."""
    draw = ImageDraw.Draw(pil_image)
    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
        b = box.cpu().numpy()
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="red", width=2)
        cls_name = class_names[label.item()] if class_names else str(label.item())
        text = f"{cls_name} {score:.2f}"
        bbox_text = draw.textbbox((b[0], b[1] - 14), text)
        draw.rectangle(bbox_text, fill="red")
        draw.text((b[0], b[1] - 14), text, fill="white")
    return pil_image


def save_gradcam_figure(orig_image, preds, heatmaps, save_path, image_id=""):
    """
    Crée et sauvegarde une figure avec :
    - colonne gauche  : image originale avec boîtes
    - colonnes suivantes : une heatmap GradCAM par détection
    """
    n_det = len(heatmaps)
    if n_det == 0:
        return

    fig, axes = plt.subplots(1, n_det + 1, figsize=(5 * (n_det + 1), 5))
    if n_det == 0:
        axes = [axes]

    # Image originale + boîtes
    img_with_boxes = draw_boxes_on_image(orig_image.copy(), preds)
    axes[0].imshow(img_with_boxes)
    axes[0].set_title(f"Détections {image_id}", fontsize=10)
    axes[0].axis("off")

    # Une heatmap par détection
    for i, (cam, label, score) in enumerate(zip(heatmaps, preds['labels'], preds['scores'])):
        if cam is None:
            axes[i + 1].set_title("No grad")
            axes[i + 1].axis("off")
            continue

        overlay = overlay_heatmap(orig_image.copy(), cam)
        # Dessiner la boîte correspondante sur l'overlay
        draw = ImageDraw.Draw(overlay)
        b = preds['boxes'][i].cpu().numpy()
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="red", width=2)

        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(
            f"Det {i} | cls={label.item()} | score={score.item():.2f}",
            fontsize=9
        )
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GradCAM] Sauvegardé : {save_path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run_gradcam(num_classes, num_images, image_path, threshold, proportion, target_class):

    project_root = Path(__file__).parent.parent
    with open("../config/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[GradCAM] Device : {device}")

    # Chargement du modèle
    model = get_model(num_classes + 1)
    model.load_state_dict(torch.load("../models/faster_rcnn.pt", map_location=device))
    model.to(device)
    # NE PAS appeler model.eval() ici — GradCAM a besoin du mode train
    # (les hooks backward ne fonctionnent pas correctement en eval sur certaines couches)

    # Initialisation du GradCAM
    gradcam = FasterRCNNGradCAM(model, target_layer_name="backbone.body.layer4")

    os.makedirs("../outputs/gradcam", exist_ok=True)
    to_tensor = transforms.Compose([transforms.ToTensor()])

    if image_path:
        # ── Mode image unique ──────────────────────────────────────────
        orig_image = Image.open(image_path).convert("RGB")
        image_tensor = to_tensor(orig_image)

        preds, heatmaps = gradcam.compute(image_tensor, device, threshold, target_class)

        if preds is not None:
            save_path = "../outputs/gradcam/gradcam_single.png"
            save_gradcam_figure(orig_image, preds, heatmaps, save_path, image_id=Path(image_path).name)
        else:
            print("[GradCAM] Rien à visualiser pour cette image.")

    else:
        # ── Mode dataset ───────────────────────────────────────────────
        my_transform = CocoToFasterRCNN()
        img_dir  = project_root / config["data"]["images"]["test"]["img_dir"]
        ann_file = project_root / config["data"]["annotations"]["test"]["ann_file"]
        dataset  = SAR_ATR_Dataset(
            root=str(img_dir), annFile=str(ann_file),
            transforms=my_transform, subset_ratio=proportion
        )

        indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

        for idx in indices:
            image_tensor, target = dataset[idx]
            orig_image = dataset._load_image(dataset.ids[idx])

            preds, heatmaps = gradcam.compute(image_tensor, device, threshold, target_class)

            if preds is not None:
                save_path = f"../outputs/gradcam/gradcam_{idx}.png"
                save_gradcam_figure(orig_image, preds, heatmaps, save_path, image_id=str(idx))
            else:
                print(f"[GradCAM] Image {idx} : aucune détection.")

    gradcam.remove_hooks()
    print("[GradCAM] Terminé. Résultats dans ../outputs/gradcam/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradCAM pour Faster RCNN ResNet50 FPN")
    parser.add_argument('--num_classes',   type=int,   default=10,   help='Nombre de classes (sans background)')
    parser.add_argument('--num_images',    type=int,   default=5,    help='Nombre d\'images à visualiser (mode dataset)')
    parser.add_argument('--image_path',    type=str,   default=None, help='Chemin vers une image spécifique')
    parser.add_argument('--threshold',     type=float, default=0.5,  help='Seuil de score pour les détections')
    parser.add_argument('--proportion',    type=float, default=1.0,  help='Proportion du dataset de test à utiliser')
    parser.add_argument('--target_class',  type=int,   default=None, help='Filtrer sur une classe spécifique (optionnel)')
    args = parser.parse_args()

    run_gradcam(
        args.num_classes,
        args.num_images,
        args.image_path,
        args.threshold,
        args.proportion,
        args.target_class
    )