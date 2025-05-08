import os
import pandas as pd
import cv2
import ast
import albumentations as A
from tqdm import tqdm

# ========== CONFIG ==========
CSV_FILE = "polar_dataset_filterd.csv"
DATASET_ROOT = "Polar_Dataset"
OUTPUT_DIR = "augmented_hyperplastic_images"
OUTPUT_CSV = "augmented_hyperplastic_metadata.csv"
AUG_PER_IMAGE = 5 #how many images created per image
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_FILE)

# Filter: hyperplastic polyps in train_set only
df_hyper = df[
    (df['SPLIT'] == 'train') &
    (df['PATHOLOGY DIAGNOSIS'].str.lower() == 'hyperplastic polyp')
].copy()

# Convert (x, y, w, h) → (x_min, y_min, x_max, y_max)
def convert_xywh_to_xyxy(box_str):
    x, y, w, h = ast.literal_eval(box_str)
    return [x, y, x + w, y + h]

df_hyper['bbox'] = df_hyper['POLYP POSITION (x,y,w,h)'].apply(convert_xywh_to_xyxy)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=20, p=0.4), #would change this
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.RandomGamma(p=0.3),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.1))

# Function to build path to image
def get_image_path(row):
    patient_id = str(row['PATIENT_ID']).strip()
    polyp_id = str(row['POLYP_ID']).strip()
    img_prefix = str(row['IMG_FILE_PREFIX']).strip()
    filename = img_prefix + ".full.png"
    return os.path.join(DATASET_ROOT, "train_set", patient_id, polyp_id, filename)

# Augment and save images + metadata
aug_records = []

print("✨ Augmenting hyperplastic polyps...")
for _, row in tqdm(df_hyper.iterrows(), total=len(df_hyper)):
    image_path = get_image_path(row)
    if not os.path.exists(image_path):
        print(f" Missing: {image_path}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Unreadable: {image_path}")
        continue

    bbox = row['bbox']
    label = ["hyperplastic"]

    for i in range(AUG_PER_IMAGE):
        try:
            aug = transform(image=img, bboxes=[bbox], category_ids=label)
            aug_img = aug["image"]
            aug_bbox = aug["bboxes"][0]

            # Create filename
            aug_filename = f"{row['IMG_FILE_PREFIX']}_aug{i}.png"
            aug_path = os.path.join(OUTPUT_DIR, aug_filename)
            cv2.imwrite(aug_path, aug_img)

            aug_records.append({
                "IMG_FILE": aug_filename,
                "PATIENT_ID": row['PATIENT_ID'],
                "POLYP_ID": row['POLYP_ID'],
                "POLYP POSITION (x_min,y_min,x_max,y_max)": aug_bbox,
                "PATHOLOGY DIAGNOSIS": "hyperplastic polyp"
            })

        except Exception as e:
            print(f"Augmentation error for {image_path}: {e}")

# Save metadata
aug_df = pd.DataFrame(aug_records)
aug_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n {len(aug_records)} augmented images saved to: {OUTPUT_DIR}")
print(f"📄 Metadata saved to: {OUTPUT_CSV}")
