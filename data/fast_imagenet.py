import h5py
import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class ImageNetDatasetH5(Dataset):
    def __init__(self, h5_path, split, transform=None, albumentations=True):
        self.h5_path = h5_path  # Path to ilsvrc2012.hdf5
        self.split = split
        self.transform = transform
        self.albumentations = albumentations
        assert os.path.exists(self.h5_path), f"ImageNet h5 file path does not exist! Given: {self.h5_path}"
        assert self.split in ["train", "val", "test"], f"split must be 'train' or 'val' or 'test'! Given: {self.split}"
        self.n_train = 1281167
        self.n_val = 50000
        self.n_test = 100000
        self.h5_data = None

    def __len__(self):
        if self.split == "train":
            return self.n_train
        elif self.split == "val":
            return self.n_val
        else:
            return self.n_test

    def __getitem__(self, idx):
        # Correct idx
        if self.split == 'val':
            idx += self.n_train
        elif self.split == 'test':
            idx += self.n_train + self.n_val
        # Read h5 file
        if self.h5_data is None:
            self.h5_data = h5py.File(self.h5_path, mode='r')
            # print([d for d in self.h5_data])
            # print(self.h5_data['targets'][0])
        # Extract info
        image = Image.open(io.BytesIO(self.h5_data['encoded_images'][idx])).convert('RGB')
        if self.transform is not None:
            if self.albumentations:
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)

        target = torch.from_numpy(self.h5_data['targets'][idx])[0].long() if self.split != 'test' else None
        return image, target


def ImageNetDataLoaders(h5_path, batch_size, workers=10, distributed=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = ImageNetDataset(h5_path, "train", train_transform)
    val_ds = ImageNetDataset(h5_path, "val", val_transform)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        sampler = None
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None),
        num_workers=workers, pin_memory=True, sampler=sampler, drop_last=True)
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)
    return train_dl, val_dl


if __name__ == "__main__":
    h5_path = "/project/rrg-bengioy-ad/codevilf/ilsvrc2012.hdf5"
    dataset = ImageNetDatasetH5(h5_path, "train")
    for x, y in dataset:
        print(x.size)