from PIL import Image
from typing import Callable, Optional
from glob import glob
import torch

from torchvision.datasets import VisionDataset

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_path=False):
        self.root = root
        self.images = []
        self.return_path = return_path
        if root[-3:] == 'txt':
            f = open(root, 'r')
            lines = f.readlines()            
            for line in lines:
                self.images.append(line.strip())
        else:
            self.images = sorted(glob(root + '/**/*.png', recursive=True))
        self.transform = transform

    def __getitem__(self, index):
        try:
            img = Image.open(self.images[index]).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            if self.return_path == False:
                return img
            else:
                return img, self.images[index]
        except Exception as e:
            print("bad image {}".format(self.images[index]))
            return self.__getitem__(0)


    def __len__(self):
        return len(self.images)

class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, root, caption, transform=None):
        self.root = root
        self.images = []
        self.caption = caption
        if root[-3:] == 'txt':
            f = open(root, 'r')
            lines = f.readlines()            
            for line in lines:
                self.images.append(line.strip())
        else:
            self.images = sorted(glob(root + '/**/*.png', recursive=True))
        self.transform = transform

    def __getitem__(self, index):
        try:
            img = Image.open(self.images[index]).convert("RGB")

            if self.transform is not None:
                img = self.transform(img)
            return img, self.caption
        except Exception as e:
            print("bad image {}".format(self.images[index]))
            return self.__getitem__(0)

    def __len__(self):
        return len(self.images)
