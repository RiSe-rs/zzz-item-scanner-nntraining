import os
import csv
import json
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# config
FONT_PATH = "inpin_hongmengti.ttf"
FONT_SIZE = 18
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (22, 22, 22)
BASE_WIDTH = 140
BASE_HEIGHT = 40
OUTPUT_DIR = "../training_data/level"
CSV_PATH = os.path.join(OUTPUT_DIR, "level_labels.csv")
MAPPING_PATH = "../mappings/level_class_mapping.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TIER_MAX_LEVEL = {
    "S": 15,
    "A": 12,
    "B": 9
}

# load font
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except OSError:
    print("font not found")


# image gen
def generate_level_images(n):
    # load mapping
    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)
    

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "class_id"])

        for class_id, data in mapping.items():
            level = int(data["level"])
            tier = data["tier"]
            max_level = TIER_MAX_LEVEL.get(tier)

            prefix = f"Lv. {level:02}"
            slash = "/"
            suffix = f"{max_level:02}"



            for sample_idx in range(n):
                width_variation = random.randint(-(BASE_WIDTH/10), (BASE_WIDTH/10))
                height_variation = random.randint(-(BASE_HEIGHT/10), (BASE_HEIGHT/10))
                img_width = BASE_WIDTH + width_variation
                img_height = BASE_HEIGHT + height_variation

                img = Image.new("RGB", (img_width, img_height), color=BG_COLOR)
                draw = ImageDraw.Draw(img)

                total_text = prefix + slash + suffix
                total_width = draw.textlength(total_text, font=font)

                # random offsets for text position
                x_offset = random.randint(-(BASE_WIDTH/10), (BASE_WIDTH/10))
                y_offset = random.randint(-(BASE_HEIGHT/10), (BASE_HEIGHT/10))

                x = (img_width - total_width) // 2 + x_offset
                y = (img_height - FONT_SIZE) // 2 + y_offset

                # draw with doubled slash to better emulate later screenshots
                draw.text((x, y), prefix, font=font, fill=TEXT_COLOR)
                x += draw.textlength(prefix, font=font)
                for dx in [0, 0.5]:
                    draw.text((x + dx, y), "/", font=font, fill=TEXT_COLOR)
                x += draw.textlength("/", font=font)
                draw.text((x + 1, y), suffix, font=font, fill=TEXT_COLOR)

                # random blur effect to emulate different image qualities
                if random.random() < 0.5:
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

                # hard resize
                if img_width != BASE_WIDTH or img_height != BASE_HEIGHT:
                    img = img.resize((140, 40), resample=Image.LANCZOS)

                # save image
                filename = f"class_{int(class_id):02}_{sample_idx:03}.png"
                img_path = os.path.join(OUTPUT_DIR, filename)
                img.save(img_path)
                writer.writerow([filename, class_id])

def main():
    generate_level_images(1000)

# === Ausführung ===
if __name__ == "__main__":
    main()
    print("Training data generated successfully.")
