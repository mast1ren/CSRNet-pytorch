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
import scipy.io as sio



from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

def get_seq_class(seq, set):
    backlight = ['DJI_0021','DJI_0022', 'DJI_0032', 'DJI_0202', 'DJI_0339', 'DJI_0340']
    # cloudy = ['DJI_0519', 'DJI_0554']
    
    # uhd = ['DJI_0332', 'DJI_0334', 'DJI_0339', 'DJI_0340', 'DJI_0342', 'DJI_0343', 'DJI_345', 'DJI_0348', 'DJI_0519', 'DJI_0544']

    fly = ['DJI_0177', 'DJI_0174', 'DJI_0022', 'DJI_0180', 'DJI_0181', 'DJI_0200', 'DJI_0544', 'DJI_0012', 'DJI_0178', 'DJI_0343', 'DJI_0185', 'DJI_0195']

    angle_90 = ['DJI_0179', 'DJI_0186', 'DJI_0189', 'DJI_0191', 'DJI_0196', 'DJI_0190']

    mid_size = ['DJI_0012', 'DJI_0013', 'DJI_0014', 'DJI_0021', 'DJI_0022', 'DJI_0026', 'DJI_0028', 'DJI_0028', 'DJI_0030', 'DJI_0028', 'DJI_0030', 'DJI_0034','DJI_0200', 'DJI_0544']

    light = 'sunny'
    bird = 'stand'
    angle = '60'
    size = 'small'
    # resolution = '4k'
    if seq in backlight:
        light = 'backlight'
    if seq in fly:
        bird = 'fly'
    if seq in angle_90:
        angle = '90'
    if seq in mid_size:
        size = 'mid'

    # if seq in uhd:
    #     resolution = 'uhd'
    
    count = 'sparse'
    loca = sio.loadmat(os.path.join('../../ds/dronebird/', set, 'ground_truth', 'GT_img'+str(seq[-3:])+'000.mat'))['locations']
    if loca.shape[0] > 150:
        count = 'crowded'
    # return light, resolution, count
    return light, angle, bird, size, count

with open('../../ds/dronebird/test.json','r') as f:
    img_paths=json.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CSRNet()

model = model.to(device)

checkpoint = torch.load('0model_best.pth.tar', map_location={'cuda:0': 'cuda:0'})
model.load_state_dict(checkpoint['state_dict'])

mae = 0
model.eval()
min_error = 1000
max_error = 0
pred = []
gt = []
preds = [[] for i in range(10)]
gts = [[] for i in range(10)]
with torch.no_grad():
    for i in range(len(img_paths)):
        # img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

        # img[0,:,:]=img[0,:,:]-92.8207477031
        # img[1,:,:]=img[1,:,:]-95.2757037428
        # img[2,:,:]=img[2,:,:]-104.877445883
        img_path = os.path.join('../../ds/dronebird', img_paths[i])
        seq = int(os.path.basename(img_path)[3:6])
        seq = 'DJI_' + str(seq).zfill(4)
        light, angle, bird, size, count = get_seq_class(seq, 'test')
        gt_path = os.path.join(os.path.dirname(img_path).replace('images', 'ground_truth'), 'GT_'+os.path.basename(img_path).replace('jpg', 'h5'))


        img = transform(Image.open(img_path).convert('RGB'))
        img = img.to(device)
        gt_file = h5py.File(gt_path,'r')
        groundtruth = np.asarray(gt_file['density'])
        output = model(img.unsqueeze(0))
        pred_e = output.detach().cpu().sum()
        gt_e = np.sum(groundtruth)
        count = 'crowded' if gt_e > 150 else 'sparse'

        if light == 'sunny':
            preds[0].append(pred_e)
            gts[0].append(gt_e)
        elif light == 'backlight':
            preds[1].append(pred_e)
            gts[1].append(gt_e)
        # else:
        #     preds[2].append(pred_e)
        #     gts[2].append(gt_e)
        if count == 'crowded':
            preds[2].append(pred_e)
            gts[2].append(gt_e)
        else:
            preds[3].append(pred_e)
            gts[3].append(gt_e)
        if angle == '60':
            preds[4].append(pred_e)
            gts[4].append(gt_e)
        else:
            preds[5].append(pred_e)
            gts[5].append(gt_e)
        if bird == 'stand':
            preds[6].append(pred_e)
            gts[6].append(gt_e)
        else:
            preds[7].append(pred_e)
            gts[7].append(gt_e)
        if size == 'small':
            preds[8].append(pred_e)
            gts[8].append(gt_e)
        else:
            preds[9].append(pred_e)
            gts[9].append(gt_e)

        error = abs(gt_e-pred_e)
        # mae += error
        pred.append(pred_e)
        gt.append(gt_e)
        # print(i,mae)
        if error > max_error:
            max_error = error
        if error < min_error:
            min_error = error

        print('\r[{:>{}}/{}], error: {:.2f} pred: {:.2f}, gt: {:.2f}, {}'.format(i+1, len(str(len(img_paths))), len(img_paths), error, pred_e, gt_e, img_path), end='')
    print('max_error: {:.2f}, min_error: {:.2f}'.format(max_error, min_error))
# print(mae/len(img_paths))

mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))
with open('test_result.txt','w') as f:
    f.write('MAE: {:.2f}, RMSE: {:.2f}\n'.format(mae, rmse))
    print ('MAE: ',mae)
    print ('RMSE: ',rmse)

    attri = ['sunny', 'backlight', 'crowded', 'sparse', '60', '90', 'stand', 'fly', 'small', 'mid']
    for i in range(10):
        if len(preds[i]) == 0:
            continue
        print('{}: MAE:{}. RMSE:{}.'.format(attri[i], mean_absolute_error(preds[i], gts[i]), np.sqrt(mean_squared_error(preds[i], gts[i]))))
        f.write('{}: MAE:{}. RMSE:{}.\n'.format(attri[i], mean_absolute_error(preds[i], gts[i]), np.sqrt(mean_squared_error(preds[i], gts[i]))))