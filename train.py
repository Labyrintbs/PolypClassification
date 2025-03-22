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
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAccuracy


import model
import train_config
from dataset import CUDAPrefetcher, CPUPrefetcher, ImageDataset, CIFAR10Dataset, PolypDataset
from utils import accuracy, load_pretrained_state_dict, load_resume_state_dict, make_directory, save_checkpoint, get_transform, \
    Summary, AverageMeter, ProgressMeter
from test import test


def main(seed):
    device = torch.device(train_config.device)
    # Fixed random number seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0

    train_prefetcher, valid_prefetcher = load_dataset(device=device)
    vgg_model, ema_vgg_model = build_model(device=device)
    criterion = define_loss(device=device)
    optimizer = define_optimizer(vgg_model)
    scheduler = define_scheduler(optimizer)

    if train_config.pretrained_model_weights_path:
        vgg_model, _, _, _, _, _ = load_pretrained_state_dict(vgg_model, train_config.pretrained_model_weights_path)
        print(f"Loaded `{train_config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    if train_config.resume_model_weights_path:
        vgg_model, ema_vgg_model, start_epoch, best_acc1, optimizer, scheduler = load_resume_state_dict(vgg_model,
                                                                                                        train_config.resume_model_weights_path,
                                                                                                        ema_vgg_model,
                                                                                                        optimizer,
                                                                                                        scheduler)
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", train_config.exp_name)
    results_dir = os.path.join("results", train_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", train_config.exp_name))

    for epoch in range(start_epoch, train_config.epochs):
        train(vgg_model, ema_vgg_model, train_prefetcher, criterion, optimizer, epoch, scaler, writer, device)
        if (epoch + 1) % train_config.val_freq == 0 or (epoch + 1) == train_config.epochs:
            acc, f1, precision, recall, auc = test(ema_vgg_model, valid_prefetcher, device)
            writer.add_scalar("Val/Accuracy", acc, epoch)
            writer.add_scalar("Val/F1", f1, epoch)
            writer.add_scalar("Val/Precision", precision, epoch)
            writer.add_scalar("Val/Recall", recall, epoch)
            writer.add_scalar("Val/AUC", auc, epoch)
            print(f"Validation at Epoch {epoch+1} — Acc: {acc:.2f}, F1: {f1:.4f}, AUC: {auc:.4f}\n")
        #acc, f1, precision, recall, auc = test(ema_vgg_model, valid_prefetcher, device)
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = acc > best_acc1
        is_last = (epoch + 1) == train_config.epochs
        best_acc1 = max(acc, best_acc1)
        save_checkpoint({"epoch": epoch + 1,
                         "best_acc1": best_acc1,
                         "state_dict": vgg_model.state_dict(),
                         "ema_state_dict": ema_vgg_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "best.pth.tar",
                        "last.pth.tar",
                        is_best,
                        is_last)


def load_dataset(
        train_image_dir: str = train_config.train_image_dir,
        valid_image_dir: str = train_config.valid_image_dir,
        resized_image_size=train_config.resized_image_size,
        crop_image_size=train_config.crop_image_size,
        device: torch.device = torch.device("cpu"),
) -> tuple:
    # Load train, test and valid datasets

    train_mean, train_std = train_config.train_mean_normalize, train_config.train_std_normalize
    val_mean, val_std = train_config.val_mean_normalize, train_config.val_std_normalize
    train_transform = get_transform('train',train_mean, train_std, train_config.resize_width, train_config.resize_height)
    valid_transform = get_transform('val',val_mean, val_std, train_config.resize_width, train_config.resize_height) 
    train_dataset = PolypDataset(train_config.train_split_dir, train_transform)
    valid_dataset = PolypDataset(train_config.val_split_dir, valid_transform)
        # train_dataset = ImageDataset(train_image_dir,
        #                             resized_image_size,
        #                             crop_image_size,
        #                             dataset_mean_normalize,
        #                             dataset_std_normalize,
        #                             "Train")
        # valid_dataset = ImageDataset(valid_image_dir,
        #                             resized_image_size,
        #                             crop_image_size,
        #                             dataset_mean_normalize,
        #                             dataset_std_normalize,
        #                             "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config.batch_size,
                                  shuffle=True,
                                  num_workers=train_config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=train_config.batch_size,
                                  shuffle=False,
                                  num_workers=train_config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    if device.type == "cuda":

        train_prefetcher = CUDAPrefetcher(train_dataloader, device)
        valid_prefetcher = CUDAPrefetcher(valid_dataloader, device)
    elif device.type == "cpu":
        train_prefetcher = CPUPrefetcher(train_dataloader)
        valid_prefetcher = CPUPrefetcher(valid_dataloader)

    return train_prefetcher, valid_prefetcher


def build_model(
        model_arch_name: str = train_config.model_arch_name,
        model_num_classes: int = train_config.model_num_classes,
        model_ema_decay: float = train_config.model_ema_decay,
        device: torch.device = torch.device("cpu"),
) -> [nn.Module, nn.Module]:
    vgg_model = model.__dict__[model_arch_name](num_classes=model_num_classes)
    vgg_model = vgg_model.to(device)

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
        (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_vgg_model = AveragedModel(vgg_model, device=device, avg_fn=ema_avg)

    return vgg_model, ema_vgg_model


def define_loss(
        loss_label_smoothing: float = train_config.loss_label_smoothing,
        device: torch.device = torch.device("cpu"),
) -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=loss_label_smoothing)
    criterion = criterion.to(device)

    return criterion


def define_optimizer(
        model: nn.Module,
        lr: float = train_config.model_lr,
        momentum: float = train_config.model_momentum,
        weight_decay: float = train_config.model_weight_decay,
) -> optim.SGD:
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)

    return optimizer


def define_scheduler(
        optimizer: optim.SGD,
        t_0: int = train_config.lr_scheduler_T_0,
        t_mult=train_config.lr_scheduler_T_mult,
        eta_min=train_config.lr_scheduler_eta_min,
) -> lr_scheduler.CosineAnnealingWarmRestarts:
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                         t_0,
                                                         t_mult,
                                                         eta_min)

    return scheduler



def train(
        model: nn.Module,
        ema_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.SGD,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device
) -> None:
    batches = len(train_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.NONE)
    # acc1 = AverageMeter("Acc", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    model.train()
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    batch_index = 0
    end = time.time()

    # Initialize binary classification metrics
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auc = BinaryAUROC().to(device)

    while batch_data is not None:
        images = batch_data["image"].to(device, non_blocking=True)
        target = batch_data["target"].to(device, non_blocking=True)

        data_time.update(time.time() - end)
        batch_size = images.size(0)

        model.zero_grad(set_to_none=True)

        with amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema_model.update_parameters(model)

        # Compute top-1 accuracy
        #top1, _ = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), batch_size)
        #acc1.update(top1[0], batch_size)

        # Compute binary metrics
        probs = torch.softmax(output, dim=1)[:, 1]  # For AUROC
        preds = torch.argmax(output, dim=1)         # For precision/recall/F1

        accuracy.update(preds, target)
        precision.update(preds, target)
        recall.update(preds, target)
        f1.update(preds, target)
        auc.update(probs, target)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % train_config.train_print_frequency == 0:
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches)
            writer.add_scalar("Train/Accuracy", accuracy.compute(),  batch_index + epoch * batches)
            writer.add_scalar("Train/Precision", precision.compute(),  batch_index + epoch * batches)
            writer.add_scalar("Train/Recall", recall.compute(),  batch_index + epoch * batches)
            writer.add_scalar("Train/F1", f1.compute(),  batch_index + epoch * batches)
            writer.add_scalar("Train/AUC", auc.compute(),  batch_index + epoch * batches)
            progress.display(batch_index)

        batch_data = train_prefetcher.next()
        batch_index += 1



if __name__ == "__main__":
    main(train_config.seed)
