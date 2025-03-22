# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import model
import test_config
from dataset import CUDAPrefetcher, CPUPrefetcher, ImageDataset, CIFAR10Dataset
from utils import load_pretrained_state_dict, accuracy, Summary, AverageMeter, ProgressMeter
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryAccuracy



def build_model(
        model_arch_name: str = test_config.model_arch_name,
        num_classes: int = test_config.model_num_classes,
        device: torch.device = torch.device("cpu"),
) -> nn.Module:
    vgg_model = model.__dict__[model_arch_name](num_classes=num_classes)
    vgg_model = vgg_model.to(device)

    return vgg_model


def load_dataset(
        test_image_dir: str = test_config.test_image_dir,
        resized_image_size=test_config.resized_image_size,
        crop_image_size=test_config.crop_image_size,
        dataset_mean_normalize=test_config.dataset_mean_normalize,
        dataset_std_normalize=test_config.dataset_std_normalize,
        device: torch.device = torch.device("cpu"),
        cifar10: bool =test_config.use_cifar10
) -> tuple:
    if cifar10:
        test_dataset = CIFAR10Dataset(root="./data", train=False)
    else:

        test_dataset = ImageDataset(test_image_dir,
                                resized_image_size,
                                crop_image_size,
                                dataset_mean_normalize,
                                dataset_std_normalize,
                                "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=test_config.batch_size,
                                 shuffle=False,
                                 num_workers=test_config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    if device.type == "cuda":
        test_prefetcher = CUDAPrefetcher(test_dataloader, device)
    elif device.type == "cpu":
        test_prefetcher = CPUPrefetcher(test_dataloader)

    return test_prefetcher


def test(
        model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        device: torch.device,
) -> tuple:
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1], prefix="Test: ")

    model.eval()
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()
    batch_index = 0
    end = time.time()

    # Init metrics
    acc_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    auc_metric = BinaryAUROC().to(device)

    with torch.no_grad():
        while batch_data is not None:
            images = batch_data["image"].to(device, non_blocking=True)
            target = batch_data["target"].to(device, non_blocking=True)

            batch_size = images.size(0)
            output = model(images)

            # For top-1 accuracy
            top1, _ = accuracy(output, target, topk=(1, 1))
            acc1.update(top1[0].item(), batch_size)

            # For binary metrics
            probs = torch.softmax(output, dim=1)[:, 1]
            preds = torch.argmax(output, dim=1)

            acc_metric.update(preds, target)
            f1_metric.update(preds, target)
            precision_metric.update(preds, target)
            recall_metric.update(preds, target)
            auc_metric.update(probs, target)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % test_config.test_print_frequency == 0:
                progress.display(batch_index)

            batch_data = data_prefetcher.next()
            batch_index += 1

    progress.display_summary()

    acc = acc_metric.compute().item()
    f1 = f1_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    auc = auc_metric.compute().item()

    return acc, f1, precision, recall, auc


def main() -> None:
    device = torch.device(test_config.device)

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Initialize the model
    vgg_model = build_model(device=device)
    vgg_model = load_pretrained_state_dict(vgg_model, test_config.model_weights_path)

    # Start the verification mode of the model.
    vgg_model.eval()

    test(vgg_model, test_prefetcher, device)


if __name__ == "__main__":
    main()
