import os
import json
import random
import numpy as np
from PIL import Image, ImageFilter


def load_annotations(ann_file):
    with open(ann_file, "r") as f:
        coco = json.load(f)

    img_dict = {img["id"]: img for img in coco["images"]}
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    return img_dict, ann_by_img, coco["categories"]


def extract_target_chip(image, bbox, margin=4):
    w_img, h_img = image.size
    x, y, w, h = bbox

    x0 = max(0, int(x - margin))
    y0 = max(0, int(y - margin))
    x1 = min(w_img, int(x + w + margin))
    y1 = min(h_img, int(y + h + margin))

    chip = image.crop((x0, y0, x1, y1))
    rel_bbox = [int(x - x0), int(y - y0), int(w), int(h)]
    return chip, rel_bbox


def extract_all_chips(images_dir, img_dict, ann_by_img, margin=4, max_chips=500):
    chips = []
    img_ids = list(ann_by_img.keys())
    random.shuffle(img_ids)

    for img_id in img_ids:
        if len(chips) >= max_chips:
            break

        info = img_dict[img_id]
        path = os.path.join(images_dir, info["file_name"])
        if not os.path.exists(path):
            continue
        image = Image.open(path).convert("RGB")

        for ann in ann_by_img[img_id]:
            chip, rel_bbox = extract_target_chip(image, ann["bbox"], margin)
            chips.append({
                "chip": chip,
                "rel_bbox": rel_bbox,
                "category_id": ann["category_id"],
                "source_id": img_id,
            })

        if len(chips) % 100 == 0:
            print(f"    {len(chips)} chips...")

    return chips


def inpaint_target(image, bbox, margin=8):
    """Replace target region with local background statistics."""
    arr = np.array(image, dtype=np.float64)
    h_img, w_img = arr.shape[:2]
    bx, by, bw, bh = [int(v) for v in bbox]

    # Clamp target bbox to image bounds
    x0 = max(0, bx)
    y0 = max(0, by)
    x1 = min(w_img, bx + bw)
    y1 = min(h_img, by + bh)
    if x0 >= x1 or y0 >= y1:
        return image

    # Expanded region for sampling background
    sx0 = max(0, x0 - margin)
    sy0 = max(0, y0 - margin)
    sx1 = min(w_img, x1 + margin)
    sy1 = min(h_img, y1 + margin)

    # Collect border pixels (expanded region minus target)
    mask = np.ones((sy1 - sy0, sx1 - sx0), dtype=bool)
    tx0 = x0 - sx0
    ty0 = y0 - sy0
    tx1 = x1 - sx0
    ty1 = y1 - sy0
    mask[ty0:ty1, tx0:tx1] = False

    region = arr[sy0:sy1, sx0:sx1]
    border_pixels = region[mask]

    if len(border_pixels) > 0:
        fill_h = y1 - y0
        fill_w = x1 - x0
        indices = np.random.randint(0, len(border_pixels), size=(fill_h, fill_w))
        arr[y0:y1, x0:x1] = border_pixels[indices]

    result = Image.fromarray(arr.astype(np.uint8))
    return result.filter(ImageFilter.GaussianBlur(radius=1))


def collect_background_tiles(images_dir, img_dict, ann_by_img, max_tiles=500):
    """Load source images with targets inpainted out, for use as background tiles."""
    tiles = []
    img_ids = list(img_dict.keys())
    random.shuffle(img_ids)

    for img_id in img_ids:
        if len(tiles) >= max_tiles:
            break

        info = img_dict[img_id]
        path = os.path.join(images_dir, info["file_name"])
        if not os.path.exists(path):
            continue

        image = Image.open(path).convert("RGB")
        anns = ann_by_img.get(img_id, [])

        for ann in anns:
            image = inpaint_target(image, ann["bbox"])

        tiles.append(np.array(image))

        if len(tiles) % 100 == 0:
            print(f"    {len(tiles)} tiles...")

    return tiles


def build_clutter_canvas(tiles, canvas_size):
    """Build realistic SAR clutter: spectral base + blended real patches."""
    if not tiles:
        return Image.fromarray(np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8))

    grays = [np.mean(t.astype(np.float64), axis=2) for t in tiles[:min(80, len(tiles))]]
    th, tw = grays[0].shape

    # Average power spectrum from real SAR data
    avg_psd = np.zeros((th, tw))
    for g in grays:
        ft = np.fft.fft2(g - g.mean())
        avg_psd += np.abs(ft) ** 2
    avg_psd /= len(grays)

    # Radial profile
    shifted = np.fft.fftshift(avg_psd)
    cy, cx = th // 2, tw // 2
    max_r = min(cy, cx)
    yy, xx = np.mgrid[:th, :tw]
    r_tile = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)

    radial = np.zeros(max_r)
    for r in range(max_r):
        m = r_tile == r
        if m.any():
            radial[r] = shifted[m].mean()
    radial = np.maximum(radial, 1e-10)

    # Build filter at canvas resolution
    cy2, cx2 = canvas_size // 2, canvas_size // 2
    yy2, xx2 = np.mgrid[:canvas_size, :canvas_size]
    r_canvas = np.sqrt((yy2 - cy2) ** 2 + (xx2 - cx2) ** 2)
    r_scaled = r_canvas * (max_r / min(cy2, cx2))
    r_idx = np.clip(r_scaled.astype(int), 0, max_r - 2)
    r_frac = np.clip(r_scaled - r_idx, 0, 1)
    freq_filter = radial[r_idx] * (1 - r_frac) + radial[np.minimum(r_idx + 1, max_r - 1)] * r_frac

    # Synthesize base field
    noise = np.random.randn(canvas_size, canvas_size)
    noise_ft = np.fft.fftshift(np.fft.fft2(noise))
    base = np.real(np.fft.ifft2(np.fft.ifftshift(noise_ft * np.sqrt(freq_filter))))

    target_mean = np.mean([g.mean() for g in grays])
    target_std = np.mean([g.std() for g in grays])
    base = (base - base.mean()) / max(base.std(), 1e-6) * target_std + target_mean

    canvas = np.clip(base, 0, 255).astype(np.uint8)
    canvas_rgb = np.stack([canvas, canvas, canvas], axis=2)
    return Image.fromarray(canvas_rgb)




