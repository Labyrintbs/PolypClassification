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
seed = 0
device = "cuda:0" 

# Model configure
model_arch_name = "vgg19"
model_num_classes = 2

# Usage of datasets

# Experiment name, easy to save weights and log files
exp_name = "2024.03.22-VGG19-PolarDataset" #"VGG11-ImageNet_1K"
# Dataset path
train_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_train"
valid_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"

# Dataset split path
train_split_dir = "./splits/train.txt"
val_split_dir = "./splits/val.txt"
test_split_dir = "./splits/test.txt"

# preprocessing args
train_mean_normalize = [0.3997, 0.3288, 0.2601]
train_std_normalize = [0.0740, 0.0677, 0.0483]
val_mean_normalize = [0.3986, 0.3282, 0.2595]
val_std_normalize = [0.0761, 0.0698, 0.0501]

resize_width = 1096
resize_height = 864

resized_image_size = 32 #256
crop_image_size = 32 #224
batch_size = 32 #128
num_workers = 4

# The address to load the pretrained model
pretrained_model_weights_path = "" #"./results/pretrained_models/VGG11-ImageNet_1K-64f6524f.pth.tar"

# Incremental training and migration training
resume_model_weights_path = ""

# Total num epochs
epochs = 10 #600
# Validation test frequency
val_freq = 2

# Loss parameters
loss_label_smoothing = 0

# Optimizer parameter
model_lr = 0.001 #0.1
model_momentum = 0.9
model_weight_decay = 2e-05
model_ema_decay = 0.99998

# Learning rate scheduler parameter
lr_scheduler_T_0 = epochs // 4
lr_scheduler_T_mult = 1
lr_scheduler_eta_min = 5e-5

# How many iterations to print the training/validate result
train_print_frequency = 10
valid_print_frequency = 10
