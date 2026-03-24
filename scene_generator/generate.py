import os
import random
import argparse
from pathlib import Path
from extract import load_annotations, extract_all_chips, collect_background_tiles
from compose import create_scene
from annotations import build_coco_annotation, save_annotation


def generate(num_scenes, canvas_size, min_targets, max_targets,
             margin, min_gap, output_dir, seed):

    random.seed(seed)
    project_root = Path(__file__).parent.parent

    images_dir = project_root / "data" / "images" / "test"
    ann_file = project_root / "data" / "annotations" / "test.json"

    print(f"Loading annotations from {ann_file}")
    img_dict, ann_by_img, categories = load_annotations(str(ann_file))
    print(f"  {len(img_dict)} images, {sum(len(v) for v in ann_by_img.values())} annotations")

    print("Collecting background tiles (inpainting targets)...")
    bg_tiles = collect_background_tiles(
        str(images_dir), img_dict, ann_by_img, max_tiles=500
    )
    print(f"  {len(bg_tiles)} background tiles collected")

    print("Extracting target chips...")
    all_chips = extract_all_chips(str(images_dir), img_dict, ann_by_img, margin=margin)
    print(f"  {len(all_chips)} target chips extracted")

    out_img_dir = os.path.join(output_dir, "images")
    out_ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    scenes_metadata = []
    print(f"\nGenerating {num_scenes} scenes ({canvas_size}x{canvas_size})...")

    for i in range(num_scenes):
        scene_img, placed_targets = create_scene(
            canvas_size, bg_tiles, all_chips,
            min_targets=min_targets, max_targets=max_targets,
            min_gap=min_gap
        )

        file_name = f"scene_{i:04d}.png"
        scene_img.save(os.path.join(out_img_dir, file_name))

        scenes_metadata.append({
            "id": i + 1,
            "file_name": file_name,
            "targets": placed_targets,
        })

        print(f"  Scene {i+1}/{num_scenes}: {len(placed_targets)} targets")

    coco_dict = build_coco_annotation(scenes_metadata, categories, canvas_size)
    ann_path = os.path.join(out_ann_dir, "scenes.json")
    save_annotation(coco_dict, ann_path)

    print(f"\nDone! Images: {out_img_dir}")
    print(f"Annotations: {ann_path}")
    print(f"Total scenes: {num_scenes}, Total annotations: {len(coco_dict['annotations'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-target SAR scenes")
    parser.add_argument("--num_scenes", type=int, default=50)
    parser.add_argument("--canvas_size", type=int, default=128)
    parser.add_argument("--min_targets", type=int, default=2)
    parser.add_argument("--max_targets", type=int, default=5)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--min_gap", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="../data/scenes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.num_scenes, args.canvas_size, args.min_targets, args.max_targets,
             args.margin, args.min_gap, args.output_dir, args.seed)
