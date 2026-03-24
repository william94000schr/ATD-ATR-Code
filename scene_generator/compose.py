import random
import numpy as np
from PIL import Image, ImageFilter


def check_overlap(existing_boxes, new_box, min_gap=4):
    nx, ny, nw, nh = new_box
    for ex, ey, ew, eh in existing_boxes:
        if (nx < ex + ew + min_gap and nx + nw + min_gap > ex and
                ny < ey + eh + min_gap and ny + nh + min_gap > ey):
            return True
    return False


def create_blend_mask(width, height, fade=6):
    """Create a feathered alpha mask that fades edges into background."""
    mask = Image.new("L", (width, height), 255)
    arr = np.array(mask, dtype=np.float64)

    # Fade each edge
    for i in range(min(fade, height // 2)):
        factor = i / fade
        arr[i, :] *= factor
        arr[height - 1 - i, :] *= factor

    for i in range(min(fade, width // 2)):
        factor = i / fade
        arr[:, i] *= factor
        arr[:, width - 1 - i] *= factor

    mask = Image.fromarray(arr.astype(np.uint8))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
    return mask


def paste_chip(canvas, chip, position):
    """Alpha-blend a chip onto the canvas with feathered edges."""
    x, y = position
    cw, ch = chip.size
    mask = create_blend_mask(cw, ch)
    canvas.paste(chip, (x, y), mask)


def create_scene(canvas_size, background_tiles, target_chips, min_targets=2,
                 max_targets=5, min_gap=4):
    from extract import build_clutter_canvas
    canvas = build_clutter_canvas(background_tiles, canvas_size)

    num_targets = random.randint(min_targets, max_targets)
    available = list(range(len(target_chips)))
    random.shuffle(available)

    placed = []
    placed_boxes = []

    for idx in available:
        if len(placed) >= num_targets:
            break

        chip_data = target_chips[idx]
        chip = chip_data["chip"]
        cw, ch = chip.size

        if cw >= canvas_size or ch >= canvas_size:
            continue

        success = False
        for _ in range(50):
            px = random.randint(0, canvas_size - cw)
            py = random.randint(0, canvas_size - ch)

            if not check_overlap(placed_boxes, (px, py, cw, ch), min_gap):
                paste_chip(canvas, chip, (px, py))

                rel = chip_data["rel_bbox"]
                scene_bbox = [px + rel[0], py + rel[1], rel[2], rel[3]]

                placed.append({
                    "bbox": scene_bbox,
                    "category_id": chip_data["category_id"],
                })
                placed_boxes.append((px, py, cw, ch))
                success = True
                break

    return canvas, placed
