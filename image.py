from pickle import FALSE
import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2


def load_data(img_path, train=True):
    img_path = os.path.join('../../nas-public-linkdata/ds/dronebird', img_path)
    gt_path = os.path.join(
        os.path.dirname(img_path).replace('images', 'ground_truth'),
        'GT_' + os.path.basename(img_path).replace('.jpg', '.h5'),
    )
    # gt_path = img_path.replace('.jpg', '.h5').replace('data', 'annotation')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if train:
        crop_size = (img.size[0] / 2, img.size[1] / 2)
        if random.randint(0, 9) <= -1:
            dx = int(random.randint(0, 1) * img.size[0] * 1.0 / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1.0 / 2)
        else:
            dx = int(random.random() * img.size[0] * 1.0 / 2)
            dy = int(random.random() * img.size[1] * 1.0 / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        # target = target[dy//2:(crop_size[1]+dy)//2, dx//2:(crop_size[0]+dx)//2]
        target = target[
            (int)(dy / 2) : (int)((crop_size[1] + dy) / 2),
            (int)(dx / 2) : (int)((crop_size[0] + dx) / 2),
        ]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # target = cv2.resize(
    #     target, (target.shape[1]//8, target.shape[0]//8), interpolation=cv2.INTER_CUBIC)*64
    target = (
        cv2.resize(
            target,
            (target.shape[1] // 4, target.shape[0] // 4),
            interpolation=cv2.INTER_CUBIC,
        )
        * 16
    )
    # print(target.shape)
    return img, target
