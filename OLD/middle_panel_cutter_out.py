import os
from PIL import Image

# Set input directory
input_dir = "FIGURES/THEBEST/"  # <-- change this to your actual directory

# Optionally, set output directory
output_dir = os.path.join(input_dir, "middle_panels")
os.makedirs(output_dir, exist_ok=True)

# Process all PNG files
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        filepath = os.path.join(input_dir, filename)
        img = Image.open(filepath)

        # Get width and height
        w, h = img.size
        panel_width = w // 3

        # Crop middle panel
        box = (panel_width, 0, 2 * panel_width, h)
        middle_panel = img.crop(box)

        # Generate output filename
        name, ext = os.path.splitext(filename)
        outname = f"{name}_middle_panel{ext}"
        outpath = os.path.join(output_dir, outname)

        # Save
        middle_panel.save(outpath)
        print(f"✔ Saved: {outpath}")

