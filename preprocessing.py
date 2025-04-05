
import pandas as pd
import os
from typing import Optional
import random
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2

def split_dataset(
    csv_path: str,
    data_dir: str,
    results_dir: str,
    val_ratio: float = 0.2,
    seed: Optional[int] = 42
) -> None:
    print(f"Spliting dataset at path {data_dir} based on file {csv_path} ")
    print(f"Train/Val splition ratio {val_ratio}, random seed {seed}")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    df_train_full = df[df["SPLIT"] == "train"].copy()
    df_test = df[df["SPLIT"] == "valid"].copy()

    df_train_full = df_train_full.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = int(len(df_train_full) * val_ratio)
    df_val = df_train_full.iloc[:val_size]
    df_train = df_train_full.iloc[val_size:]

    min_class_size = df_train["PATHOLOGY DIAGNOSIS"].value_counts().min()

    balanced_df_list = []
    for label, group_df in df_train.groupby("PATHOLOGY DIAGNOSIS"):
        balanced_df_list.append(group_df.sample(n=min_class_size, random_state=seed))

    df_balanced_train = pd.concat(balanced_df_list).sample(frac=1, random_state=seed).reset_index(drop=True)


    def get_img_path(row):
        patient_id = row['PATIENT_ID']
        polyp_id = row['POLYP_ID']
        image_prefix = row["IMG_FILE_PREFIX"]

        if row['SPLIT'] == 'train':                           
            img_folder = os.path.join(data_dir, "train_set", patient_id, polyp_id)
        elif row['SPLIT'] == 'valid':
            img_folder = os.path.join(data_dir, "validation_set", patient_id) 
        else:
            raise FileExistsError(f"This folder {img_folder} doesn't exist")
        if os.path.exists(img_folder):
            for img_file in os.listdir(img_folder):
                if img_file.startswith(image_prefix): 
                    img_path = os.path.join(img_folder, img_file)
                    return img_path
        return None

    for name, subset in[("train", df_train), ("val", df_val), ("test", df_test), ("balanced_train", df_balanced_train)]:
        samples = []
        for _, row in subset.iterrows():
            path = get_img_path(row)
            label = row["PATHOLOGY DIAGNOSIS"]  
            if path:
                samples.append((path, label))

        txt_path = os.path.join(results_dir, f"{name}.txt")
        with open(txt_path, "w") as f:
            for path, label in samples:
                f.write(f"{path}\t{label}\n")

        print(f"{name}: {len(samples)} samples saved to {txt_path}")



def analyze_image_statistics(image_paths: list[str]) -> tuple:
    """
    Compute image dataset statistics, including size, color channels, per-channel mean/std, 
    brightness, and contrast.

    Args:
    - train_paths: list[str] - List of image file paths

    Returns:
    - stats: dict - A dictionary containing computed statistics for all images
    """
    stats = defaultdict(list)

    for img_path in image_paths:
        img = Image.open(img_path)
        img_cv = cv2.imread(img_path)

        w, h = img.size
        stats["widths"].append(w)
        stats["heights"].append(h)

        color_mode = "Grayscale" if len(img.getbands()) == 1 else "RGB"
        stats["color_modes"].append(color_mode)


        img_arr = np.array(img).astype(np.float32) / 255.0

        if img_arr.ndim == 2:  # Grayscale
            stats["mean_R"].append(np.mean(img_arr))
            stats["std_R"].append(np.std(img_arr))
        else:  # RGB
            for i, c in enumerate(['R', 'G', 'B']):
                stats[f"mean_{c}"].append(np.mean(img_arr[:, :, i]))
                stats[f"std_{c}"].append(np.std(img_arr[:, :, i]))

        img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
        stats["brightness"].append(np.mean(img_yuv[:, :, 0]))
        stats["contrast"].append(np.std(img_yuv[:, :, 0]))

    # Aggregate mean/std for each channel
    mean_std_summary = {}
    for key in stats:
        if key.startswith("mean_") or key.startswith("std_") or key in ["brightness", "contrast"]:
            mean_std_summary[key] = {
                "mean": np.mean(stats[key]),
                "std": np.std(stats[key])
            }

    return stats, mean_std_summary

def load_paths_and_labels(txt_path: str):
    paths, labels = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            path, label = line.strip().split("\t")
            paths.append(path)
            labels.append(label)
    return paths, labels


if __name__ ==  "__main__":
    print("Start preprocessing!")
    csv_path="/Users/tuboshu/Documents/2024/M2/IMA_Project/PolypClassification/polar_dataset_filterd.csv"
    data_dir="/Users/tuboshu/Downloads/Polar_Dataset"
    results_dir="./splits"
    split_dataset(csv_path, data_dir, results_dir)
    train_paths, train_labels = load_paths_and_labels("splits/train.txt")
    stats, summary = analyze_image_statistics(train_paths)

    for c in ["mean_R", "mean_G", "mean_B"]:
        print(f"Train {c}: {summary[c]['mean']:.4f} ± {summary[c]['std']:.4f}")
    
    test_paths, test_labels = load_paths_and_labels("splits/test.txt")
    stats, summary = analyze_image_statistics(test_paths)

    for c in ["mean_R", "mean_G", "mean_B"]:
        print(f"Test {c}: {summary[c]['mean']:.4f} ± {summary[c]['std']:.4f}")
    
    val_paths, val_labels = load_paths_and_labels("splits/val.txt")
    #stats, summary = analyze_image_statistics(val_paths)

    for c in ["mean_R", "mean_G", "mean_B"]:
        print(f"Val {c}: {summary[c]['mean']:.4f} ± {summary[c]['std']:.4f}")


    print("Finish preprocessing!")