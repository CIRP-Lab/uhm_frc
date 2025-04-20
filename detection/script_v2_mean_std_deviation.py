import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import supervision as sv
from inference import get_model

# Configuration
APIKEY = "IcD1jh3rgXGTLxbkBapc"
MOD_ID = "treecnt/4"
INPUT_DIR  = "./original_images"
OUTPUT_DIR = "./processed_images"
TILE_SIZE  = 640

def split_image(image: Image.Image, tile_size: int):
    width, height = image.size
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            tiles.append((image.crop(box), (x, y)))
    return tiles

def reconstruct_image(tiles, original_size):
    canvas = Image.new('RGB', original_size)
    for tile, (x, y) in tiles:
        canvas.paste(tile, (x, y))
    return canvas

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
try:
    model = get_model(model_id=MOD_ID, api_key=APIKEY)
except Exception as e:
    print(f"Failed to load model: {e}")
    raise SystemExit

box_annotator   = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Prepare a larger font
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 100)
except IOError:
    font = ImageFont.load_default()

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"\n--- Processing {filename} ---")
    img_path  = os.path.join(INPUT_DIR, filename)
    orig      = Image.open(img_path).convert("RGB")
    tiles     = split_image(orig, TILE_SIZE)

    processed_tiles = []
    counts_info     = []  # (x, y, count)

    # 1) Infer & annotate each tile
    for tile, (x, y) in tiles:
        try:
            infer_res = model.infer(tile)[0]
            dets      = sv.Detections.from_inference(infer_res)
            count     = len(dets.xyxy)
            counts_info.append((x, y, count))

            a = box_annotator.annotate(scene=tile, detections=dets)
            a = label_annotator.annotate(scene=a,     detections=dets)
            processed_tiles.append((a, (x, y)))
        except Exception as e:
            print(f"Inference failed at ({x},{y}): {e}")
            counts_info.append((x, y, 0))
            processed_tiles.append((tile, (x, y)))

    # 2) Reconstruct & save detections image
    det_image = reconstruct_image(processed_tiles, orig.size)
    det_path  = os.path.join(OUTPUT_DIR, f"detections_{filename}")
    det_image.save(det_path)
    print(f"→ Saved detections image: {det_path}")

    # 3) Compute and print mean & standard deviation
    counts    = [c for _, _, c in counts_info]
    mean_cnt  = np.mean(counts)
    std_cnt   = np.std(counts)
    print(f"→ Tiles: {len(counts)}, Mean detections/tile: {mean_cnt:.2f}, Std.dev: {std_cnt:.2f}")

    # 4) Compute padded dimensions (next multiples of TILE_SIZE)
    orig_w, orig_h = orig.size
    pad_w = math.ceil(orig_w / TILE_SIZE) * TILE_SIZE
    pad_h = math.ceil(orig_h / TILE_SIZE) * TILE_SIZE

    # 5) Create padded canvas (black) and paste original
    grid_img = Image.new('RGB', (pad_w, pad_h), color="black")
    grid_img.paste(orig, (0, 0))
    draw = ImageDraw.Draw(grid_img)

    # 6) Draw grid + centered counts
    for x, y, count in counts_info:
        draw.rectangle(
            [(x, y), (x + TILE_SIZE, y + TILE_SIZE)],
            outline="red",
            width=2
        )
        txt  = str(count)
        bbox = draw.textbbox((0, 0), txt, font=font)
        w    = bbox[2] - bbox[0]
        h    = bbox[3] - bbox[1]
        cx   = x + (TILE_SIZE - w) // 2
        cy   = y + (TILE_SIZE - h) // 2
        draw.text((cx, cy), txt, fill="yellow", font=font)

    # 7) Save grid image
    grid_path = os.path.join(OUTPUT_DIR, f"grid_{filename}")
    grid_img.save(grid_path)
    print(f"→ Saved padded grid image: {grid_path}  (size: {pad_w}×{pad_h})")
