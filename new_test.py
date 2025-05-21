#new_test.py
from sklearn.metrics import classification_report, confusion_matrix

import os
import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAccuracy
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import model
import train_config
from dataset import CUDAPrefetcher, CPUPrefetcher, PolypDataset
from utils import get_transform
from torchvision.models import vgg19, VGG19_Weights
import pandas as pd

# --------------------------------------
# Step 1: Find the best F1 epoch
# --------------------------------------
def find_best_f1_epoch(log_dir):
    print(f"📊 Reading TensorBoard logs from: {log_dir}")
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if "Val/F1" not in event_acc.Tags()["scalars"]:
        raise ValueError("❌ F1 score not found in logs!")

    f1_events = event_acc.Scalars("Val/F1")
    best_event = max(f1_events, key=lambda x: x.value)
    best_epoch = best_event.step
    best_f1 = best_event.value

    print(f"✅ Best F1 score: {best_f1:.4f} at epoch {best_epoch}")
    return best_epoch

# --------------------------------------
# Step 2: Main evaluation function
# --------------------------------------
def main(checkpoint_path):
    device = torch.device(train_config.device)

    print(f"📦 Loading model checkpoint from: {checkpoint_path}")
    model_arch = model.__dict__[train_config.model_arch_name](num_classes=train_config.model_num_classes)
    model_arch.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    model_arch = model_arch.to(device)
    model_arch.eval()

    # C. Load test dataset
    test_mean, test_std = train_config.train_mean_normalize, train_config.train_std_normalize
    if train_config.model_pretrain_torch:
        vgg_weights = VGG19_Weights.DEFAULT
        test_transform = vgg_weights.transforms()
    else:
        test_transform = get_transform('val', test_mean, test_std, train_config.resize_width, train_config.resize_height)
    
    #test_dataset = PolypDataset(os.path.expanduser("./splits/test.txt"), test_transform)
    test_dataset = PolypDataset(os.path.expanduser("./splits/val.txt"), test_transform)

    test_loader = DataLoader(test_dataset,
                             batch_size=train_config.batch_size,
                             shuffle=False,
                             num_workers=train_config.num_workers,
                             pin_memory=True,
                             drop_last=False,
                             persistent_workers=True)

    test_prefetcher = (
        CUDAPrefetcher(test_loader, device) if device.type == "cuda" else CPUPrefetcher(test_loader)
    )

    # D. Set up metrics
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)

    # E. Run evaluation
    print("🚀 Running evaluation on test set...")
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    all_preds = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        while batch_data is not None:
            images = batch_data["image"].to(device)
            targets = batch_data["target"].to(device)
            paths = batch_data["path"] 

            outputs = model_arch(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_paths.extend(paths)

            accuracy.update(preds, targets)
            precision.update(preds, targets)
            recall.update(preds, targets)
            f1.update(preds, targets)
            auc.update(probs, targets)

            batch_data = test_prefetcher.next()

    # F. Final results
    print("\n🎯 Final Test Results:")
    print(f"Accuracy : {accuracy.compute():.4f}")
    print(f"Precision: {precision.compute():.4f}")
    print(f"Recall   : {recall.compute():.4f}")
    print(f"F1 Score : {f1.compute():.4f}")
    print(f"AUC      : {auc.compute():.4f}")

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    print("\n Classification Report:")
    print(classification_report(all_targets, all_preds, digits=4))

    print(" Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

    idx_to_class = {0: "adenoma", 1: "hyperplastic polyp"}
    df = pd.DataFrame({
        "path": all_paths,
        "target": all_targets,
        "pred": all_preds
    })
    df["target_name"] = df["target"].map(idx_to_class)
    df["pred_name"] = df["pred"].map(idx_to_class)
    errors = df[df["target"] != df["pred"]]
    csv_file = "error/1stage/val/2024.05.12-VGG19-torch-BalancedTrain1Stage-Batch-128-seed-4_valid.csv"
    errors.to_csv(csv_file, index=False)

    print(f"\n❌ Number of incorrect predictions: {len(errors)}")
    print(f"📁 Misclassified sample paths saved to {csv_file}")


if __name__ == "__main__":
    checkpoint_path = '~/IMA_project/PolypClassification/samples/2024.05.12-VGG19-torch-BalancedTrain1Stage-Batch-128-seed-4/epoch_120.pth.tar'
    checkpoint_path = os.path.expanduser(checkpoint_path)

    main(checkpoint_path)

