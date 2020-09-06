import cv2
import torch
import numpy as np
from model import FAN
from transform import *
from utils import *
import matplotlib.pyplot as plt
from dataset import Dataset
import torch.utils.data as data
from torch.backends import cudnn
import hello


#1. initialize model and weights
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint = './checkpoint/models/best_checkpoint.pth.tar' # checkpoint path
model = FAN(3,81, 4)
state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict, False)
model.eval()
model.to(device)

imgdirs_test_commomset = ['D:/dataset/ibug/']
# imgdirs_test_commomset = ['D:/dataset/300W/300WAugment/300Wtest/']

testset = Dataset(imgdirs_test_commomset, 'test', 'test', 0, 128, 3)
test_loader = data.DataLoader(testset,batch_size=1,num_workers=0,pin_memory=True)
data_iter = iter(test_loader)
error = 0
error1 = 0
for i in range(135):
    images, targets, kps, tforms = next(data_iter)
    img = images.to(device)
    img_part = images.squeeze(0).numpy()
    img_part = np.swapaxes(img_part, 0, 1)
    img_part = np.swapaxes(img_part, 1, 2)
    images_flip = torch.from_numpy(images.cpu().numpy()[:, :, :, ::-1].copy())  # 左右翻转
    img1 = images_flip.to(device)
    # plt.imshow(img_part)
    # plt.show()
    with torch.no_grad():
        out = model(img)
        out1 = model(img1)
        out1 = flip_channels(out1.cpu())
        out1 = shuffle_channels_for_horizontal_flipping(out1)
        out = (out1.cpu() + out.cpu()) / 2
        # out2 = get_preds(out)
        # pred = out2.squeeze(0).numpy()
        rmse, pred_pts = rmse_batch(out, kps, tforms)
        heatmap = out[:,0:68,:,:].detach().cpu().numpy()
        cut_size = 7
        sub_kpts = hello.get_subpixel_from_kpts(pred_pts, heatmap, cut_size)
        sub_kpts = torch.from_numpy(sub_kpts)
        sub_kpts = sub_kpts.view(-1, 68, 2)
        rmse2 = per_image_rmse(sub_kpts, kps, tforms)
        error = error + rmse
        error1 = error1 + rmse2
        print('rmse is: ', rmse, rmse2)
    # show_preds(img_part, pred)
print('mean error is: ', error/135, error1/135)