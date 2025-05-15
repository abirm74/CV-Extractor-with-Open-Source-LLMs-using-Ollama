import os
import img2pdf
from PIL import Image

screenshots_folder = "data/scannedcvs/screenshots"
output_folder = "data/scanned_cvs//"
os.makedirs(output_folder, exist_ok=True)

# Group screenshots by filename prefix (e.g., cv13_pg1.png, cv13_pg2.png)
grouped = {}
for img in sorted(os.listdir(screenshots_folder)):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        base_name = "_".join(img.split("_")[:1])  # e.g., 'cv13'
        grouped.setdefault(base_name, []).append(os.path.join(screenshots_folder, img))

# Convert each group into one PDF
for base_name, img_paths in grouped.items():
    pdf_path = os.path.join(output_folder, f"{base_name}_scanned.pdf")
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(img_paths))
    print(f"✅ Created scanned PDF: {pdf_path}")
