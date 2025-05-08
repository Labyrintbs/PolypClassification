import pandas as pd
import shutil
import os

# Step 1: Load CSVs
df_orig = pd.read_csv("polar_dataset_filterd.csv")  # adjust if not tab-separated
df_aug = pd.read_csv("augmented_hyperplastic_metadata.csv")

# Step 2: Sample 1800 rows
df_aug_sampled = df_aug.sample(n=1800, random_state=42).copy()

# Step 3: Convert bounding boxes from (x_min, y_min, x_max, y_max) → (x, y, w, h)
def convert_bbox(bbox_str):
    # Remove brackets and split
    clean_str = bbox_str.strip().replace('[', '').replace(']', '')
    x_min, y_min, x_max, y_max = map(float, clean_str.split(','))
    x = int(x_min)
    y = int(y_min)
    w = int(x_max - x_min)
    h = int(y_max - y_min)
    return f"{x},{y},{w},{h}"


df_aug_sampled['POLYP POSITION (x,y,w,h)'] = df_aug_sampled['POLYP POSITION (x_min,y_min,x_max,y_max)'].apply(convert_bbox)

# Rename image file column to match original
df_aug_sampled.rename(columns={
    'IMG_FILE': 'IMG_FILE_PREFIX',
}, inplace=True)

# Add missing columns
for col in ['SIZE', 'BBOX_AREA', 'SPLIT']:
    df_aug_sampled[col] = pd.NA

# Add augmentation flag
df_orig['AUGMENTED'] = False
df_aug_sampled['AUGMENTED'] = True

# Step 4: Combine the datasets
df_combined = pd.concat([df_orig, df_aug_sampled[df_orig.columns]], ignore_index=True)

# Step 5: Move images into the correct folder
src_folder = "augmented_hyperplastic_images"
dst_folder = os.path.join("Polar_Dataset", "augmented")
os.makedirs(dst_folder, exist_ok=True)

for img_file in df_aug_sampled['IMG_FILE_PREFIX']:
    src_path = os.path.join(src_folder, img_file)
    dst_path = os.path.join(dst_folder, img_file)
    shutil.copy2(src_path, dst_path)

# Step 6: Save the combined dataset
df_combined.to_csv("polar_dataset_with_augmented.csv", index=False)
