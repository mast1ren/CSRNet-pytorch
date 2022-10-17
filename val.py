#!/usr/bin/env python
# coding: utf-8


import h5py
import PIL.Image as Image
import numpy as np
import json
import torchvision.transforms.functional as F
from image import *
from model import CSRNet
import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error



from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

with open('test.json','r') as f:
    img_paths=json.load(f)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = CSRNet()

model = model.to(device)

checkpoint = torch.load('0model_best.pth.tar', map_location={'cuda:0': 'cuda:2'})
model.load_state_dict(checkpoint['state_dict'])

mae = 0
model.eval()
min_error = 1000
max_error = 0
pred = []
gt = []
with torch.no_grad():
    for i in range(len(img_paths)):
        # img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

        # img[0,:,:]=img[0,:,:]-92.8207477031
        # img[1,:,:]=img[1,:,:]-95.2757037428
        # img[2,:,:]=img[2,:,:]-104.877445883
        img = transform(Image.open(img_paths[i]).convert('RGB'))
        img = img.to(device)
        gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('data','annotation'),'r')
        groundtruth = np.asarray(gt_file['density'])
        output = model(img.unsqueeze(0))
        pred_e = output.detach().cpu().sum()
        gt_e = np.sum(groundtruth)
        error = abs(gt_e-pred_e)
        # mae += error
        pred.append(pred_e)
        gt.append(gt_e)
        # print(i,mae)
        if error > max_error:
            max_error = error
        if error < min_error:
            min_error = error

        print('\r[{:>{}}/{}], error: {:.2f} pred: {:.2f}, gt: {:.2f}, {}'.format(i+1, len(str(len(img_paths))), len(img_paths), error, pred_e, gt_e, img_paths[i]), end='')
    print('max_error: {:.2f}, min_error: {:.2f}'.format(max_error, min_error))
# print(mae/len(img_paths))

mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

print ('MAE: ',mae)
print ('RMSE: ',rmse)

