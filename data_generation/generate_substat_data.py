import os
import re
import csv
import json
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# config
FONT_PATH = "../inpin_hongmengti.ttf"
FONT_SIZE = 18
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (22, 22, 22)
ROLL_COLOR = (255, 175, 42)
BASE_WIDTH = 412
BASE_HEIGHT = 40
PADDING = 15
OUTPUT_DIR = "../training_data/substat"
CSV_PATH = os.path.join(OUTPUT_DIR, "substat_labels.csv")
MAPPING_PATH = "../mappings/substat_class_mapping.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load font
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except OSError:
    print("font not found")


# image gen
def generate_substat_images(n):
    # load mapping
    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)
    

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "class_id"])

        for class_id, data in mapping.items():
            rolls = data["rolls"]
            value = data["value"]
            stat_name = data["stat_name"]
            # remove _flat and _percent fom ATK, DEF and HP
            stat_name = re.split(r"_", stat_name, maxsplit=1)[0]

            for sample_idx in range(n):
                rand_font_size = max(10, int(FONT_SIZE*(random.uniform(0.8, 1.2))))

                try:
                    font = ImageFont.truetype(FONT_PATH, rand_font_size)
                except OSError:
                    print("font not found, using default")
                    font = ImageFont.load_default()

                width_variation = random.randint(int(-(BASE_WIDTH/10)), int((BASE_WIDTH/10)))
                height_variation = random.randint(int(-(BASE_HEIGHT/10)), int((BASE_HEIGHT/10)))
                img_width = BASE_WIDTH + width_variation
                img_height = BASE_HEIGHT + height_variation

                img = Image.new("RGB", (img_width, img_height), color=BG_COLOR)
                draw = ImageDraw.Draw(img)

                # random offsets for text position
                x_offset = random.randint(int(-(BASE_WIDTH/20)), int((BASE_WIDTH/20)))
                y_offset = random.randint(int(-(BASE_HEIGHT/10)), int((BASE_HEIGHT/10)))

                x = PADDING + x_offset
                y = (img_height - rand_font_size) // 2 + y_offset

                # add 1-3 space charaters after stat name and append rolls
                spaces = " " * random.randint(1, 3)
                stat_and_spaces = stat_name+spaces

                # draw stat on left side
                draw.text((x, y), stat_name, font=font, fill=TEXT_COLOR)
                # draw rolls after stat name
                rolls_x = x + draw.textlength(stat_and_spaces, font=font)
                draw.text((rolls_x, y), rolls, font=font, fill=ROLL_COLOR)

                # draw value on right side
                text_width_value = draw.textlength(value, font=font)
                x = img_width - text_width_value - PADDING + (x_offset/2)
                draw.text((x, y), value, font=font, fill=TEXT_COLOR)

                # random blur effect to emulate different image qualities
                if random.random() < 0.5:
                    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

                # hard resize
                if img_width != BASE_WIDTH or img_height != BASE_HEIGHT:
                    img = img.resize((BASE_WIDTH, BASE_HEIGHT), resample=Image.LANCZOS)

                # save image
                filename = f"class_{int(class_id):02}_{sample_idx:03}.png"
                img_path = os.path.join(OUTPUT_DIR, filename)
                img.save(img_path)
                writer.writerow([filename, class_id])

def main():
    generate_substat_images(1000)

if __name__ == "__main__":
    main()
    print("Training data generated successfully.")
