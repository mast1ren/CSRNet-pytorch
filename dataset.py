import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
# import matplotlib.pyplot as plt
# import cv2
# import torchvision.transforms.functional as F
from torchvision import datasets, transforms


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4, downsample=4):
        if train:
            root = root * 4
        random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downsample = downsample

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        img, target = load_data(img_path, self.train)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

# if __name__ == '__main__':
#     train_list = ["E:/dronebird/dronebird/train/DJI_0014/data/000000.jpg"]
#     train_loader = torch.utils.data.DataLoader(
#         listDataset(train_list, shuffle=True,
#                             transform=transforms.Compose([
#                                 transforms.ToTensor(), transforms.Normalize(
#                                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                             ]), train=True, seen=0, batch_size=1, num_workers=4),
#         batch_size=1)

#     for i, (img, target) in enumerate(train_loader):
#         plt.figure(1)
#         plt.imshow(img.squeeze(0).permute(1, 2, 0))
#         # plt.imshow(img.permute(1, 2, 0))
#         plt.figure(2)
#         plt.imshow(target.permute(1, 2, 0))
#         plt.show()
