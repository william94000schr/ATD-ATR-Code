import json


def build_coco_annotation(scenes_metadata, categories, canvas_size):
    images = []
    annotations = []
    ann_id = 1

    for scene in scenes_metadata:
        images.append({
            "file_name": scene["file_name"],
            "height": canvas_size,
            "width": canvas_size,
            "id": scene["id"],
        })

        for target in scene["targets"]:
            bbox = target["bbox"]
            annotations.append({
                "area": int(bbox[2] * bbox[3]),
                "iscrowd": 0,
                "image_id": scene["id"],
                "bbox": [int(b) for b in bbox],
                "category_id": target["category_id"],
                "id": ann_id,
                "ignore": 0,
                "segmentation": [],
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def save_annotation(coco_dict, output_path):
    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=2)
