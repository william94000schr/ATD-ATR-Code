import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def _draw_boxes(draw, boxes, labels, scores=None, color="red", class_names=None, text_below=False):
    for i, (box, label) in enumerate(zip(boxes, labels)):
        b = box.cpu().numpy()
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=color, width=2)
        name = (
            class_names.get(str(label.item()), str(label.item()))
            if class_names
            else str(label.item())
        )
        text = f"{name} {scores[i]:.2f}" if scores is not None else name
        y_text = b[3] if text_below else b[1] - 14
        bbox = draw.textbbox((b[0], y_text), text)
        draw.rectangle(bbox, fill=color)
        draw.text((b[0], y_text), text, fill="white")


def save_prediction(orig_image, preds, save_path, ground_truth=None, class_names=None):
    """Save image with predictions (red) and optional ground truth (green)."""
    image = orig_image.copy()
    draw = ImageDraw.Draw(image)

    if ground_truth is not None:
        _draw_boxes(
            draw,
            ground_truth["boxes"],
            ground_truth["labels"],
            color="green",
            class_names=class_names,
            text_below=True,
        )

    _draw_boxes(
        draw, preds["boxes"], preds["labels"], preds["scores"], color="red", class_names=class_names
    )
    image.save(save_path)


def _overlay_heatmap(pil_image, cam, alpha=0.45):
    orig_w, orig_h = pil_image.size
    cam_resized = (
        np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )

    heatmap_rgb = (cm.get_cmap("jet")(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    orig_np = np.array(pil_image.convert("RGB"))
    blended = (orig_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def save_gradcam(orig_image, preds, heatmaps, save_path, ground_truth=None, class_names=None):
    """Save a figure with one column per detection showing its GradCAM overlay.

    The first column shows the original image with all predicted boxes (red) and
    optional ground-truth boxes (green). Each subsequent column shows the GradCAM
    heatmap overlaid on the image for one detection.
    """
    n_det = len(heatmaps)
    if n_det == 0:
        return

    fig, axes = plt.subplots(1, n_det + 1, figsize=(5 * (n_det + 1), 5))
    if n_det + 1 == 1:
        axes = [axes]

    # Left column: original image with all boxes
    img_left = orig_image.copy()
    draw = ImageDraw.Draw(img_left)
    if ground_truth is not None:
        _draw_boxes(
            draw,
            ground_truth["boxes"],
            ground_truth["labels"],
            color="green",
            class_names=class_names,
            text_below=True,
        )
    _draw_boxes(
        draw, preds["boxes"], preds["labels"], preds["scores"], color="red", class_names=class_names
    )
    axes[0].imshow(img_left)
    axes[0].set_title("Detections", fontsize=10)
    axes[0].axis("off")

    # One GradCAM overlay per detection
    for i, (cam, label, score) in enumerate(zip(heatmaps, preds["labels"], preds["scores"])):
        if cam is None:
            axes[i + 1].set_title("No gradient")
            axes[i + 1].axis("off")
            continue

        overlay = _overlay_heatmap(orig_image.copy(), cam)
        draw = ImageDraw.Draw(overlay)
        b = preds["boxes"][i].cpu().numpy()
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="red", width=2)

        name = (
            class_names.get(str(label.item()), str(label.item()))
            if class_names
            else str(label.item())
        )
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"{name} | {score.item():.2f}", fontsize=9)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
