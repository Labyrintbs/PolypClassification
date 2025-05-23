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
import queue
import sys
import threading
from glob import glob

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import find_classes
from torchvision.transforms import TrivialAugmentWide

from imgproc import image_to_tensor
from torchvision.datasets import CIFAR10
import os

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper that ensures images are tensors."""
    def __init__(self, root: str, train: bool, transform=None):
        self.dataset = CIFAR10(root=root, train=train, download=True)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 normalization
        ])

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform(image)  # Ensure it's a tensor
        return {"image": image, "target": target}

    def __len__(self):
        return len(self.dataset)


class PolypDataset(Dataset):
    def __init__(self, txt_file: str, transform=None):
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {"adenoma": 0, "hyperplastic polyp": 1}
        with open(txt_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                path, label = line.strip().split("\t")
                if label not in self.class_to_idx:
                    print(f"Skipping unknown label: {label}")
                    continue
                self.image_paths.append(path)
                self.labels.append(self.class_to_idx[label])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return {"image": image, "target": label,  "path": self.image_paths[idx]}
    
class ImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        images_dir (str): Train/Valid dataset address.
        resized_image_size (int): Resized image size.
        crop_image_size (int): Crop image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(
            self,
            images_dir: str,
            resized_image_size: int,
            crop_image_size: int,
            mean_normalize: tuple = None,
            std_normalize: tuple = None,
            mode: str = "train",
    ) -> None:
        super(ImageDataset, self).__init__()
        if mean_normalize is None:
            mean_normalize = (0.485, 0.456, 0.406)
        if std_normalize is None:
            std_normalize = (0.229, 0.224, 0.225)
        # Iterate over all image paths
        self.images_file_path = glob(f"{images_dir}/*/*")
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(images_dir)
        self.crop_image_size = crop_image_size
        self.resized_image_size = resized_image_size
        self.mean_normalize = mean_normalize
        self.std_normalize = std_normalize
        self.mode = mode
        self.delimiter = delimiter

        if self.mode == "Train":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop((self.resized_image_size, self.resized_image_size)),
                TrivialAugmentWide(),
                transforms.RandomRotation([0, 270]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        elif self.mode == "Valid" or self.mode == "Test":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.Resize((self.resized_image_size, self.resized_image_size)),
                transforms.CenterCrop((self.crop_image_size, self.crop_image_size)),
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(self.mean_normalize, self.std_normalize)
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:
        images_dir, images_name = self.images_file_path[batch_index].split(self.delimiter)[-2:]
        # Read a batch of image data
        if images_name.split(".")[-1].lower() in IMG_EXTENSIONS:
            image = cv2.imread(self.images_file_path[batch_index])
            target = self.class_to_idx[images_dir]
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # OpenCV convert PIL
        image = Image.fromarray(image)

        # Data preprocess
        image = self.pre_transform(image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = image_to_tensor(image, False, False)

        # Data postprocess
        tensor = self.post_transform(tensor)

        return {"image": tensor, "target": target}

    def __len__(self) -> int:
        return len(self.images_file_path)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
