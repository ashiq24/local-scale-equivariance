"""Implements MNIST DataModule."""
import os
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from .mnist import MultiDigitMNIST
from sklearn.model_selection import train_test_split


class MNISTDataModule():
    def __init__(self,
                 data_dir,
                 batch_size,
                 num_workers=1,
                 image_size=64,
                 n_digits=2,
                 s=None,
                 concat_type='horizontal',
                 val_ratio=0.2,
                 image_channels=1,
                 image_format='TENSOR'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.s = s
        self.concat_type = concat_type
        self.val_ratio = val_ratio
        self.image_channels = image_channels
        self.image_format = image_format
        self.n_digits = n_digits
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        # download the data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self):
        # transform
        if self.image_format == 'TENSOR':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.image_format == 'PIL':
            transform = transforms.ToTensor()
        else:
            raise ValueError(f'Image format {self.image_format} not found')

        # load the data
        train_data = MultiDigitMNIST(self.data_dir,
                                      train=True,
                                      transform=transform,
                                      img_size=self.image_size,
                                      scales = self.s,
                                      concat_type=self.concat_type,
                                      image_channels=self.image_channels,
                                      n_digits=self.n_digits)
        test_data = MultiDigitMNIST(self.data_dir,
                                     train=False,
                                     transform=transform,
                                     img_size=self.image_size,
                                     scales=self.s,
                                     concat_type=self.concat_type,
                                     image_channels=self.image_channels,
                                     n_digits=self.n_digits)

        # split the data
        num_val = int(len(train_data) * self.val_ratio)
        num_train = len(train_data) - num_val
        self.train_data, self.val_data = random_split(
            train_data, [num_train, num_val])
        self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
