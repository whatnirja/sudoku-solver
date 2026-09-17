from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random

font_paths = [
  "/System/Library/Fonts/Supplemental/Arial.ttf",
  "/System/Library/Fonts/Supplemental/Verdana.ttf",
  "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
  "/System/Library/Fonts/Supplemental/Courier New.ttf",
  "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
]
output_dir = "printed_digits"

for digit in range(10):
  digit_dir = os.path.join(output_dir, str(digit))
  os.makedirs(digit_dir, exist_ok=True)

  for font_path in font_paths:
    font_name = os.path.basename(font_path).replace(".ttf", "")

    for i in range(20):
      image = Image.new("L", (28, 28), color=0)
      draw = ImageDraw.Draw(image)

      font_size = random.randint(18, 24)
      font = ImageFont.truetype(font_path, font_size)

      text = str(digit)
      bbox = draw.textbbox((0, 0), text, font=font)
      text_w = bbox[2] - bbox[0]
      text_h = bbox[3] - bbox[1]
      x = (28 - text_w) // 2 - bbox[0]
      y = (28 - text_h) // 2 - bbox[1]

      draw.text((x, y), text, fill=255, font=font)

      angle = random.uniform(-10, 10)
      image = image.rotate(angle, fillcolor=0)

      save_path = os.path.join(digit_dir, f"{font_name}_{i}.png")

      if random.random() < 0.5:
        blur_radius = random.uniform(0.3, 1.0)
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))

      image.save(save_path)

print("Done generating printed digits.")