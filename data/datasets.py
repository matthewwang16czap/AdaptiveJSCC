from PIL import Image
import random
import os
import numpy as np
from glob import glob
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import cv2
from .letterbox import LetterBox


class LetterboxImageDataset(Dataset):
    def __init__(self, dirs, image_dims, max_samples=None):
        """
        dirs: paths to images
        image_dims: (C, H, W)
        """
        self.paths = []
        for d in dirs:
            self.paths += glob(os.path.join(d, "*.png"))
            self.paths += glob(os.path.join(d, "*.jpg"))
        self.paths.sort()
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        assert len(self.paths) > 0, f"No images found in {dirs}"
        C, H, W = image_dims
        self.letterbox = LetterBox(
            new_shape=(H, W),
            auto=False,
            scale_fill=False,
            scaleup=True,
            center=True,
            padding_value=0,
            interpolation=cv2.INTER_CUBIC,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])  # BGR, uint8
        if img is None:
            raise RuntimeError(f"Failed to load image: {self.paths[idx]}")
        img_tensor, valid = self.letterbox(image=img)  # BGR, uint8
        img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).contiguous()
        img_tensor = img_tensor.float() / 255.0  # [0,1]
        valid = torch.from_numpy(valid).float()
        return img_tensor, valid


class RandomResizedCropImageDataset(Dataset):
    def __init__(self, dirs, image_dims, train, max_samples=None):
        """
        image_dims: (C,H,W)
        train: True = Random crop, False = Center crop or resize
        """
        self.paths = []
        for d in dirs:
            self.paths += glob(os.path.join(d, "*.png"))
            self.paths += glob(os.path.join(d, "*.jpg"))
        self.paths.sort()
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        _, H, W = image_dims
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((H, W), scale=(0.7, 1.0)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(min(H, W)),  # preserves aspect ratio
                    transforms.CenterCrop((H, W)),  # exact final size
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor


def worker_init_fn_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(name, data_dirs, config, train):
    """
    name: "DIV2K", "KODAK", or folder dataset name
    """
    name = name.upper()
    if config.dataset_type.lower() == "letterbox":
        return LetterboxImageDataset(
            dirs=data_dirs,
            image_dims=config.image_dims,
            max_samples=config.max_test_samples if not train else None,
        )
    return RandomResizedCropImageDataset(
        dirs=data_dirs,
        image_dims=config.image_dims,
        train=train,
        max_samples=config.max_test_samples if not train else None,
    )


def get_loader(config, rank=None, world_size=None, num_workers=None):
    train_dataset = get_dataset(
        name=config.trainset,
        data_dirs=config.train_data_dir,
        config=config,
        train=True,
    )
    test_dataset = get_dataset(
        name=config.testset,
        data_dirs=config.test_data_dir,
        config=config,
        train=False,
    )
    # Sampler (DDP)
    if rank is not None and world_size is not None:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True
    if num_workers is None:
        num_workers = min(4, os.cpu_count() // world_size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn_seed,
        persistent_workers=False,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2,
    )
    return train_loader, test_loader, train_sampler, test_sampler
