#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:52:44 2018

@author: xiang
"""
import random
import torch
import torchvision
import torch.utils.data as data 
import cv2
import numpy as np
import pickle
from utils import get_imglists, rotatepoints, procrustes, draw_gaussian, enlarge_box, flippoints, get_gtbox, show_image, loadFromPts
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

heatmap_size = 64
boundary_num = 13
sigma1 = 2
sigma2 = 3

boundary_keys = ['chin', 'leb', 'reb', 'bon', 'breath', 'lue', 'lle', 'rue', 'rle', 'usul', 'lsul', 'usll', 'lsll']

interp_points_num = {
    'chin':   120,
    'leb':    32,
    'reb':    32,
    'bon':    32,
    'breath': 25,
    'lue':    25,
    'lle':    25,
    'rue':    25,
    'rle':    25,
    'usul':   32,
    'lsul':   32,
    'usll':   32,
    'lsll':   32
}

dataset_pdb_numbins = {
    '300W': 9,
    'AFLW': 17,
    'COFW': 7,
    'WFLW': 13
}
dataset_size = {
    '300W': {
        'train':            3148,
        'common_subset':    554,
        'challenge_subset': 135,
        'fullset':          689,
        '300W_testset':     600,
        'COFW68':           507  # 该数据集用于300W数据集上训练模型的测试
    },
    'AFLW': {
        'train':            20000,
        'test':             24386,
        'frontal':          1314
    },
    'COFW': {
        'train':            1345,
        'test':             507
    },
    'WFLW': {
        'train':            7500,
        'test':             2500,
        'pose':             326,
        'expression':       314,
        'illumination':     698,
        'makeup':           206,
        'occlusion':        736,
        'blur':             773
    }
}

kp_num = {
    '300W': 68,
    'AFLW': 19,
    'COFW': 29,
    'WFLW': 98
}

point_num_per_boundary = {
    '300W': [17., 5., 5., 4., 5., 4., 4., 4., 4., 7., 5., 5., 7.],
    'AFLW': [1.,  3., 3., 1., 2., 3., 3., 3., 3., 3., 3., 3., 3.],
    'COFW': [1.,  3., 3., 1., 3., 3., 3., 3., 3., 3., 1., 1., 3.],
    'WFLW': [33., 9., 9., 4., 5., 5., 5., 5., 5., 7., 5., 5., 7.]
}

boundary_special = {  # 有些边界线条使用的关键点和其他边界形成不连续交集，特殊处理
    'lle':  ['300W', 'COFW', 'WFLW'],
    'rle':  ['300W', 'COFW', 'WFLW'],
    'usll': ['300W', 'WFLW'],
    'lsll': ['300W', 'COFW', 'WFLW']
}

duplicate_point = {  # 需要重复使用的关键点的序号，从0开始计数
    '300W': {
        'lle':  36,
        'rle':  42,
        'usll': 60,
        'lsll': 48
    },
    'COFW': {
        'lle':  13,
        'rle':  17,
        'lsll': 21
    },
    'WFLW': {
        'lle':  60,
        'rle':  68,
        'usll': 88,
        'lsll': 76
    }
}

point_range = {  # notice: this is 'range', the later number pluses 1; the order is boundary order; index starts from 0
    '300W': [
        [0, 17],  [17, 22], [22, 27], [27, 31], [31, 36],
        [36, 40], [39, 42], [42, 46], [45, 48], [48, 55],
        [60, 65], [64, 68], [54, 60]
    ],
    'AFLW': [
        [0, 1],   [1, 4],   [4, 7],   [7, 8],   [8, 10],
        [10, 13], [10, 13], [13, 16], [13, 16], [16, 19],
        [16, 19], [16, 19], [16, 19]
    ],
    'COFW': [
        [0, 1],   [1, 4],   [5, 8],   [9, 10],  [10, 13],
        [13, 16], [15, 17], [17, 20], [19, 21], [21, 24],
        [25, 26], [26, 27], [23, 25]
    ],
    'WFLW': [
        [0, 33],  [33, 38], [42, 47], [51, 55], [55, 60],
        [60, 65], [64, 68], [68, 73], [72, 76], [76, 83],
        [88, 93], [92, 96], [82, 88]
    ]
}

flip_relation = {
    '300W': [
        [0, 16],  [1, 15],  [2, 14],  [3, 13],  [4, 12],  [5, 11],
        [6, 10],  [7, 9],   [8, 8],   [9, 7],   [10, 6],  [11, 5],
        [12, 4],  [13, 3],  [14, 2],  [15, 1],  [16, 0],  [17, 26],
        [18, 25], [19, 24], [20, 23], [21, 22], [22, 21], [23, 20],
        [24, 19], [25, 18], [26, 17], [27, 27], [28, 28], [29, 29],
        [30, 30], [31, 35], [32, 34], [33, 33], [34, 32], [35, 31],
        [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
        [42, 39], [43, 38], [44, 37], [45, 36], [46, 41], [47, 40],
        [48, 54], [49, 53], [50, 52], [51, 51], [52, 50], [53, 49],
        [54, 48], [55, 59], [56, 58], [57, 57], [58, 56], [59, 55],
        [60, 64], [61, 63], [62, 62], [63, 61], [64, 60], [65, 67],
        [66, 66], [67, 65]
    ],
    'AFLW': [
        [0, 0],   [1, 6],   [2, 5],   [3, 4],   [4, 3],   [5, 2],
        [6, 1],   [7, 7],   [8, 9],   [9, 8],   [10, 15], [11, 14],
        [12, 13], [13, 12], [14, 11], [15, 10], [16, 18], [17, 17],
        [18, 16]
    ],
    'COFW': [
        [0, 0],   [1, 7],   [2, 6],   [3, 5],   [4, 8],   [5, 3],
        [6, 2],   [7, 1],   [8, 4],   [9, 9],   [10, 12], [11, 11],
        [12, 10], [13, 19], [14, 18], [15, 17], [16, 20], [17, 15],
        [18, 14], [19, 13], [20, 16], [21, 23], [22, 22], [23, 21],
        [24, 24], [25, 25], [26, 26], [27, 28], [28, 27]
    ],
    'WFLW': [
        [0, 32],  [1, 31],  [2, 30],  [3, 29],  [4, 28],  [5, 27],
        [6, 26],  [7, 25],  [8, 24],  [9, 23],  [10, 22], [11, 21],
        [12, 20], [13, 19], [14, 18], [15, 17], [16, 16], [17, 15],
        [18, 14], [19, 13], [20, 12], [21, 11], [22, 10], [23, 9],
        [24, 8],  [25, 7],  [26, 6],  [27, 5],  [28, 4],  [29, 3],
        [30, 2],  [31, 1],  [32, 0],  [33, 46], [34, 45], [35, 44],
        [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],
        [42, 37], [43, 36], [44, 35], [45, 34], [46, 33], [47, 41],
        [48, 40], [49, 39], [50, 38], [51, 51], [52, 52], [53, 53],
        [54, 54], [55, 59], [56, 58], [57, 57], [58, 56], [59, 55],
        [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75],
        [66, 74], [67, 73], [68, 64], [69, 63], [70, 62], [71, 61],
        [72, 60], [73, 67], [74, 66], [75, 65], [76, 82], [77, 81],
        [78, 80], [79, 79], [80, 78], [81, 77], [82, 76], [83, 87],
        [84, 86], [85, 85], [86, 84], [87, 83], [88, 92], [89, 91],
        [90, 90], [91, 89], [92, 88], [93, 95], [94, 94], [95, 93],
        [96, 97], [97, 96]
    ]
}

lo_eye_corner_index_x = {'300W': 72, 'AFLW': 20, 'COFW': 26, 'WFLW': 120}
lo_eye_corner_index_y = {'300W': 73, 'AFLW': 21, 'COFW': 27, 'WFLW': 121}
ro_eye_corner_index_x = {'300W': 90, 'AFLW': 30, 'COFW': 38, 'WFLW': 144}
ro_eye_corner_index_y = {'300W': 91, 'AFLW': 31, 'COFW': 39, 'WFLW': 145}
l_eye_center_index_x = {'300W': [72, 74, 76, 78, 80, 82], 'AFLW': 22, 'COFW': 54, 'WFLW': 192}
l_eye_center_index_y = {'300W': [73, 75, 77, 79, 81, 83], 'AFLW': 23, 'COFW': 55, 'WFLW': 193}
r_eye_center_index_x = {'300W': [84, 86, 88, 90, 92, 94], 'AFLW': 28, 'COFW': 56, 'WFLW': 194}
r_eye_center_index_y = {'300W': [85, 87, 89, 91, 93, 95], 'AFLW': 29, 'COFW': 57, 'WFLW': 195}

nparts = {  # [chin, brow, nose, eyes, mouth], totally 5 parts
    '300W': [
        [0, 17], [17, 27], [27, 36], [36, 48], [48, 68]
    ],
    'WFLW': [
        [0, 33], [33, 51], [51, 60],  [60, 76], [76, 96]
    ]
}
def watch_gray_heatmap(gt_heatmap):
    heatmap_sum = gt_heatmap[0]
    plt.imshow(heatmap_sum)
    plt.show()

def get_gt_heatmap(dataset, gt_coords):
    coord_x, coord_y, gt_heatmap = [], [], []
    for index in range(boundary_num):
        gt_heatmap.append(np.ones((128, 128)))
        gt_heatmap[index].tolist()
    boundary_x = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
                  'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}
    boundary_y = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
                  'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}
    points = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
              'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}

    for boundary_index in range(boundary_num):#boundary_index：0-12
        for kp_index in range(point_range[dataset][boundary_index][0],point_range[dataset][boundary_index][1]):
            boundary_x[boundary_keys[boundary_index]].append(gt_coords[kp_index,0])
            boundary_y[boundary_keys[boundary_index]].append(gt_coords[kp_index,1])
        if boundary_keys[boundary_index] in boundary_special.keys() and dataset in boundary_special[boundary_keys[boundary_index]]:
            tmp = gt_coords[duplicate_point[dataset][boundary_keys[boundary_index]],0]
            boundary_x[boundary_keys[boundary_index]].append(gt_coords[duplicate_point[dataset][boundary_keys[boundary_index]],0])
            boundary_y[boundary_keys[boundary_index]].append(gt_coords[duplicate_point[dataset][boundary_keys[boundary_index]],1])

    for k_index, k in enumerate(boundary_keys):
        if point_num_per_boundary[dataset][k_index] >= 2.:
            if len(boundary_x[k]) == len(set(boundary_x[k])) or len(boundary_y[k]) == len(set(boundary_y[k])):
                # print(k_index)
                points[k].append(boundary_x[k])
                points[k].append(boundary_y[k])
                res = splprep(points[k], s=0.0, k=1)
                u_new = np.linspace(res[1].min(), res[1].max(), interp_points_num[k])
                boundary_x[k], boundary_y[k] = splev(u_new, res[0], der=0) #利用B样条和它的导数进行插值，

    for index, k in enumerate(boundary_keys):
        if point_num_per_boundary[dataset][index] >= 2.: #边界包含的点的数量大于等于2
            for i in range(len(boundary_x[k]) - 1): #i 从0 到边界包含点的数量-1
                # 起点到终点划线，元素值设为0
                cv2.line(gt_heatmap[index], (int(boundary_x[k][i]), int(boundary_y[k][i])),
                         (int(boundary_x[k][i+1]), int(boundary_y[k][i+1])), 0)
        else:
            cv2.circle(gt_heatmap[index], (int(boundary_x[k][0]), int(boundary_y[k][0])), 2, 0, -1)
        gt_heatmap[index] = np.uint8(gt_heatmap[index])
        # 利用distanceTransform计算像素距离矩阵，离边界越近值越接近于0，相差一个像素距离为1
        gt_heatmap[index] = cv2.distanceTransform(gt_heatmap[index], cv2.DIST_L2, 5)
        gt_heatmap[index] = np.float32(np.array(gt_heatmap[index]))
        gt_heatmap[index] = gt_heatmap[index].reshape(128*128) #拉成一列，像素距离小于6的，使用指数进行概率转换
        #将与边界线距离小于3* sigma的点，使用指数处理/2 * sigma * sigma计算概率值
        (gt_heatmap[index])[(gt_heatmap[index]) < 6] = \
            np.exp(-(gt_heatmap[index])[(gt_heatmap[index]) < 6] *
                   (gt_heatmap[index])[(gt_heatmap[index]) < 6] /(2. * sigma2 * sigma2))
        (gt_heatmap[index])[(gt_heatmap[index]) >= 6] = 0.001
        gt_heatmap[index] = gt_heatmap[index].reshape([128, 128])
        gt_heatmap_tmp = gt_heatmap[0]
    return np.array(gt_heatmap)




class Dataset(data.Dataset): # torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override these methods
    def __init__(self, imgdirs, phase, attr, rotate, res=128, gamma=3, target_type='heatmap'):
        
        self.imglists = get_imglists(imgdirs)
        assert phase in ['train', 'test'], 'Only support train and test'
        self.phase = phase
        self.r = rotate
        self.res = res
        assert target_type in ['heatmap','landmarks'], 'Only support heatmap regression and landmarks regression'
        self.target_type = target_type
        self.gamma = gamma
        self.transform = torchvision.transforms.ToTensor() # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
                                                           # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    def __len__(self): # __len__ so that len(dataset) returns the size of the dataset.
        return len(self.imglists)
    
    def __getitem__(self, i): # __getitem__ to support the indexing such that dataset[i] can be used to get ith sample
        # 1. load image and kps
        image = cv2.imread(self.imglists[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,c = image.shape

        kps_path = self.imglists[i][:-4]+'.pts'
        kps = loadFromPts(kps_path)
        
        # 2. data augmentation
        if self.phase == 'train':
            # rotate
            angle = random.randint(-self.r, self.r)
            r_kps = rotatepoints(kps,[w/2,h/2],angle) # 逆时针旋转angle度
            
            # norm kps to [0,res] 
            #旋转之后的kps的方框
            left = np.min(r_kps[:, 0]) # 所有行的第0列
            right = np.max(r_kps[:, 0])
            top = np.min(r_kps[:, 1]) # 所有行的第1列
            bot = np.max(r_kps[:, 1])
            
            r_kps -=[left, top] # 坐标转换到[0,-]
            r_kps[:,0] *= self.res/(right-left) # 坐标转换到[0,res]
            r_kps[:,1] *= self.res/(bot-top)
            
            # scale
            s = random.uniform(0.9, 1.2) # uniform()方法将随机生成浮点数，它在 [x, y) 范围内
            # make scale around center 
            dx = (1-s)*self.res * 0.5 # res*0.5-s*res*0.5 缩放前的中心-缩放后的中心
            s_kps = r_kps*s + [dx,dx]
            
            # translation
            dx = random.uniform(-self.res*0.1, self.res*0.1)
            dy = random.uniform(-self.res*0.1, self.res*0.1)
            t_kps = s_kps + [dx,dy]
            
            # procrustes analysis 从两组关键点间分析出变换矩阵用于图像的变换
            d, Z, tform = procrustes(t_kps, kps) # a dict specifying the rotation, translation and scaling that maps X --> Y
            M = np.zeros([2,3],dtype=np.float32)
            M[:2,:2] = tform['rotation'].T * tform['scale']
            M[:,2] = tform['translation']
            img = cv2.warpAffine(image,M,(self.res,self.res)) # 仿射变换 将图像按照关键点变换
            new_kps = np.dot(kps,tform['rotation']) * tform['scale'] + tform['translation']

            
        
        else:
            # enlarge box 
            box = get_gtbox(kps)
            box = enlarge_box(box,0.05)
            xmin, ymin, xmax, ymax = box

            
            src = np.array([[xmin,ymin],[xmin,ymax],[xmax,ymin],[xmax,ymax]])
            dst = np.array([[0,0],[0,self.res-1],[self.res-1,0],[self.res-1,self.res-1]])
            
            # procrustes analysis
            d, Z, tform = procrustes(dst, src)
            M = np.zeros([2,3],dtype=np.float32)
            M[:2,:2] = tform['rotation'].T * tform['scale']
            M[:,2] = tform['translation']
            img = cv2.warpAffine(image,M,(self.res,self.res))

            new_kps = np.dot(kps,tform['rotation']) * tform['scale'] + tform['translation']


        if self.phase == 'train':
            # flip
            if random.random() > 0.5:
                img = img[:, ::-1] # 左右翻转
                new_kps = flippoints(new_kps, self.res)
            
            # resize
            if random.random() > 0.8:
                new_res = int(self.res*0.75)
                img = cv2.resize(img,(new_res,new_res))
                img = cv2.resize(img,(self.res,self.res))
                
        if self.target_type == 'heatmap':
            num_points = kps.shape[0]+13
            new_kps = new_kps.astype(np.int32)
            target = np.ones([num_points,self.res,self.res])
            for n in range(68):
                target[n] = draw_gaussian(target[n], new_kps[n], sigma=self.gamma)
                    # 构造训练的heatmap的标签
            # for i in range(81):
            #     target_tmp = target[i, :, :]
            #     plt.imshow(target_tmp)
            #     plt.show()
            target_tmp = get_gt_heatmap('300W', new_kps)
            target[68:81, :, :] = target_tmp[:, :, :]
            # for i in range(67, 81):
            #     target_tmp = target[i, :, :]
            #     plt.imshow(target_tmp)
            #     plt.show()
            target = torch.from_numpy(target).float() # 将numpy格式转换成torch.tensor格式
        else:
            target = torch.from_numpy(new_kps).float() # 回归landmark



        # img to tensor
        img = self.transform(img.copy()) # transforms.ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to 
                                        #a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        if self.phase == 'train':
            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
        
        
        
        return img, target, torch.from_numpy(new_kps), tform
