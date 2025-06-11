import os
import csv
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# config
FONT_PATH = "../inpin_hongmengti.ttf"
FONT_SIZE = 22
TEXT_COLOR = (255, 255, 255)
OUTLINE_COLOR = (0, 0, 0)
BASE_WIDTH = 280
BASE_HEIGHT = 60
PADDING = 15
OUTPUT_DIR = "../training_data/setname"
CSV_PATH = os.path.join(OUTPUT_DIR, "setname_labels.csv")
WORDS = string.ascii_letters+"-.'"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2-13 letter string word, or small chance for &
def generate_word():
    if random.random() < 0.005:
        return "&"
    length = random.randint(2, 14)
    return ''.join(random.choices(WORDS, k=length))

# make noisy grey background
def noisy_background(img_width, img_height):
    base_color = random.randint(20, 40)
    img = Image.new("RGB", (img_width, img_height), (base_color, base_color, base_color))
    pixels = img.load()
    for y in range(img_height):
        for x in range(img_width):
            noise = random.randint(-10, 10)
            r = max(0, min(255, pixels[x, y][0] + noise))
            g = max(0, min(255, pixels[x, y][1] + noise))
            b = max(0, min(255, pixels[x, y][2] + noise))
            pixels[x, y] = (r, g, b)
    return img

# draw text with outline
def draw_text(draw, pos, text, font):
    x, y = pos
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=OUTLINE_COLOR)
    draw.text((x, y), text, font=font, fill=TEXT_COLOR)

# generate setname with 1-5 words
def generate_setname():
    words = [generate_word() for _ in range(random.randint(1, 5))]
    return ' '.join(words)

# draw noisy rectangles in second line if empty
def heavy_noise(draw, y):
    for _ in range(50):
        x0 = random.randint(0, BASE_WIDTH-PADDING)
        x1 = x0 + random.randint(5, 30)
        y0 = y + random.randint(0, 10)
        y1 = y0 + random.randint(5, 30)
        color = tuple(random.randint(30, 220) for _ in range(3))
        draw.rectangle([x0, y0, x1, y1], fill=color)

# image gen
def generate_setname_images(n):
    attempts = 0
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        i = 0
        while i < n:
            attempts += 1
            
            rand_font_size = max(10, int(FONT_SIZE*(random.uniform(0.8, 1.2))))
            try:
                font = ImageFont.truetype(FONT_PATH, rand_font_size)
            except OSError:
                font = ImageFont.load_default()

            width_variation = random.randint(int(-(BASE_WIDTH/10)), int((BASE_WIDTH/10)))
            height_variation = random.randint(int(-(BASE_HEIGHT/10)), int((BASE_HEIGHT/10)))
            img_width = BASE_WIDTH + width_variation
            img_height = BASE_HEIGHT + height_variation

            img = noisy_background(img_width, img_height)
            draw = ImageDraw.Draw(img)

            # generate setname with slot
            setname = generate_setname()
            slot = f"[{random.randint(1, 6)}]"
            full_text = f"{setname} {slot}"

            # line wrapping
            words = full_text.split()
            line1 = full_text
            line2 = ""

            while draw.textlength(line1, font=font) > img_width - PADDING and len(words) > 1:
                line2 = ' '.join([words[-1]] + line2.split())
                words = words[:-1]
                line1 = ' '.join(words)

            # check if words fit into lines, skip attempt if not
            if draw.textlength(line1, font=font) > img_width - PADDING:
                continue
            if line2 and draw.textlength(line2, font=font) > img_width - 20:
                continue

            # random offsets for text position
            y_offset1 = random.randint(int(-(img_height/20)), int(img_height/20))
            y_offset2 = random.randint(int(-(img_height/20)), int(img_height/20))
            x_offset = random.randint(int(-(PADDING/4)), int(PADDING/4))
            y1 = (img_height/2 - rand_font_size) // 2 + y_offset1
            y2 = y1+rand_font_size+ y_offset2

            # draw text
            draw_text(draw, (x_offset, y1), line1, font)
            if line2:
                draw_text(draw, (x_offset, y2), line2, font)
            else:
                # if second line is empty, draw noise to simulate game icons
                heavy_noise(draw, y=y1+rand_font_size+img_height//10)

            # random blur effect to emulate different image qualities
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

            # hard resize
            if img_width != BASE_WIDTH or img_height != BASE_HEIGHT:
                img = img.resize((BASE_WIDTH, BASE_HEIGHT), Image.LANCZOS)
            
            # save image
            filename = f"set_{i:05}.png"
            img.save(os.path.join(OUTPUT_DIR, filename))
            label = f"{line1} {line2}".strip()
            writer.writerow([filename, label])
            i += 1
    return attempts

def main():
    attempts = generate_setname_images(10000)
    print(f"Training data generated succesfully with {attempts} attempts.")

if __name__ == "__main__":
    main()
