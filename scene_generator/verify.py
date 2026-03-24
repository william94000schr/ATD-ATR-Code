import os
import json
from pathlib import Path
from PIL import Image, ImageDraw


def verify():
    project_root = Path(__file__).parent.parent
    scenes_dir = project_root / "data" / "scenes"
    img_dir = scenes_dir / "images"
    ann_file = scenes_dir / "annotations" / "scenes.json"

    # Check annotation file exists
    assert ann_file.exists(), f"Annotation file not found: {ann_file}"
    with open(ann_file, "r") as f:
        coco = json.load(f)

    # Validate structure
    for key in ["images", "annotations", "categories"]:
        assert key in coco, f"Missing key: {key}"

    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")

    valid_cats = {c["id"] for c in coco["categories"]}
    img_ids = {img["id"] for img in coco["images"]}
    errors = 0

    for ann in coco["annotations"]:
        # Valid image reference
        if ann["image_id"] not in img_ids:
            print(f"  ERROR: annotation {ann['id']} references missing image {ann['image_id']}")
            errors += 1

        # Valid category
        if ann["category_id"] not in valid_cats:
            print(f"  ERROR: annotation {ann['id']} has invalid category {ann['category_id']}")
            errors += 1

    for img_info in coco["images"]:
        path = img_dir / img_info["file_name"]
        if not path.exists():
            print(f"  ERROR: image file missing: {path}")
            errors += 1
            continue

        img = Image.open(path)
        if img.size != (img_info["width"], img_info["height"]):
            print(f"  ERROR: size mismatch for {img_info['file_name']}")
            errors += 1

    # Check bboxes within bounds
    img_sizes = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    for ann in coco["annotations"]:
        w, h = img_sizes.get(ann["image_id"], (0, 0))
        bx, by, bw, bh = ann["bbox"]
        if bx < 0 or by < 0 or bx + bw > w or by + bh > h:
            print(f"  WARNING: bbox out of bounds in image {ann['image_id']}, ann {ann['id']}")
            errors += 1

    if errors == 0:
        print("\nAll checks passed!")
    else:
        print(f"\n{errors} error(s) found.")

    # Generate sample visualizations
    samples_dir = scenes_dir / "samples"
    os.makedirs(samples_dir, exist_ok=True)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_info in coco["images"][:3]:
        path = img_dir / img_info["file_name"]
        if not path.exists():
            continue

        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for ann in ann_by_img.get(img_info["id"], []):
            bx, by, bw, bh = ann["bbox"]
            draw.rectangle([(bx, by), (bx + bw, by + bh)], outline="lime", width=1)
            label = cat_names.get(ann["category_id"], str(ann["category_id"]))
            text_bbox = draw.textbbox((bx, by - 12), label)
            draw.rectangle(text_bbox, fill="lime")
            draw.text((bx, by - 12), label, fill="black")

        out_path = samples_dir / f"sample_{img_info['id']:04d}.png"
        img.save(out_path)
        print(f"Sample saved: {out_path}")


if __name__ == "__main__":
    verify()
