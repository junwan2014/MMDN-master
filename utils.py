#https://github.com/bearpaw/pytorch-pose/blob/master/pose/utils/imutils.py
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches # 生成图形(matplotlib.patches)
import math

point_range = {  # notice: this is 'range', the later number pluses 1; the order is boundary order; index starts from 0
    '300W': [
        [0, 17],  [17, 22], [22, 27], [27, 31], [31, 36],
        [36, 40], [40, 42], [42, 46], [46, 48], [48, 55],
        [60, 65], [65, 68], [55, 60]
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

def get_imglists(img_dirs):
    imglists = []
    for i in range(len(img_dirs)):
        imglists += glob.glob(img_dirs[i] + "*.jpg") # glob查找符合规则的文件
        imglists += glob.glob(img_dirs[i] + "*.png")
    return imglists

def get_gtbox(landmarks):
    
    bbox = np.zeros(4)
    bbox[0] = min(landmarks[:,0])
    bbox[1] = min(landmarks[:,1])
    bbox[2] = max(landmarks[:,0])
    bbox[3] = max(landmarks[:,1])
    
    return bbox


def loadFromPts(filename):
    landmarks = np.genfromtxt(filename, skip_header=3, skip_footer=1)
    landmarks = landmarks - 1
    return landmarks

def saveToPts(filename, landmarks):
    pts = landmarks + 1
    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(filename, pts, delimiter=' ', header=header, footer='}', fmt='%.3f', comments='')

def get_preds(scores): # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    tmp = scores.view(scores.size(0), scores.size(1), -1)
    maxval, idx = torch.max(tmp, 2)
    # maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) 
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) 


    return preds

def get_preds2(scores, pred_pts, gt_pts): # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    #转换成numpy数据
    pred = np.zeros((1,68,2))
    scores = scores.numpy()
    dataset = '300W'
    radius=1
    for i  in range(68):
        landmark_part = scores[0, i, :, :]
        start_x = int(pred_pts[0, i, 0] - radius)
        end_x = int(pred_pts[0, i, 0] + radius)
        if start_x <= 0:
            start_x = 0
            end_x = start_x + 2*radius
        if end_x > 127:
            start_x = 127 - 2*radius
            end_x = 127
        start_y = int(pred_pts[0, i, 1] - radius)
        end_y = int(pred_pts[0, i, 1] + radius)
        if start_y <= 0:
            start_y = 0
            end_y = start_y + 2*radius
        if end_y > 127:
            start_y = 127 - 2*radius
            end_y = 127
        landmark_part_tmp = landmark_part[start_y:end_y + 1, start_x:end_x + 1]
        landmark_part_tmp = landmark_part_tmp.reshape(1, -1)
        max_val = np.argmax(landmark_part_tmp, 1) + 1
        pred[0, i, 0] = (max_val - 1) % (2*radius+1) + start_x
        pred[0, i, 1] = np.floor((max_val - 1) / (2*radius+1)) + start_y
    return pred

def get_preds1(scores, pred_pts, gt_pts): # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    #转换成numpy数据
    pred = np.zeros((1,68,2))
    scores = scores.numpy()
    dataset = '300W'
    boundary_num = 13
    radius=1
    for boundary_index in range(boundary_num):#boundary_index：0-12
        boundary_part = scores[0, 68+boundary_index, :, :]
        for kp_index in range(point_range[dataset][boundary_index][0],point_range[dataset][boundary_index][1]):
            start_x = int(pred_pts[0, kp_index, 0] - radius)
            end_x = int(pred_pts[0, kp_index, 0] + radius)
            if start_x <= 0:
                start_x = 0
                end_x = start_x + 2*radius
            if end_x > 127:
                start_x = 127 - 2*radius
                end_x = 127
            start_y = int(pred_pts[0, kp_index, 1] - radius)
            end_y = int(pred_pts[0, kp_index, 1] + radius)
            if start_y <= 0:
                start_y = 0
                end_y = start_y + 2*radius
            if end_y > 127:
                start_y = 127 - 2*radius
                end_y = 127
            boundary_part_tmp = boundary_part[start_y:end_y + 1, start_x:end_x + 1]
            boundary_part_tmp = boundary_part_tmp.reshape(1, -1)
            max_val = np.argmin(boundary_part_tmp, 1) + 1
            pred[0, kp_index, 0] = (max_val - 1) % (2*radius+1) + start_x
            pred[0, kp_index, 1] = np.floor((max_val - 1) / (2*radius+1)) + start_y
    return pred



def get_new_output(output): # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert output.dim() == 4, 'Score maps should be 4-dim'
    output_landmark = output[:, 0:68, :, :]
    output_boundary = output[:, 68:81, :, :]
    dataset='300W'
    new_out = torch.zeros((output.size(0), 68, 128, 128))
    for i in range(13):
        start_index = point_range[dataset][i][0]
        end_index = point_range[dataset][i][1]
        landmark_part = output_landmark[:, start_index:end_index, :, :]
        boundary_part = output_boundary[:, (i):(i+1), :, :]
        boundary_part = boundary_part.repeat(1, end_index-start_index, 1, 1)
        new_part = landmark_part.mul(boundary_part)
        new_out[:, start_index:end_index, :, :] = new_part
    return new_out

def get_new_output1(output): # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert output.dim() == 4, 'Score maps should be 4-dim'
    output_landmark = output[:, 0:68, :, :]
    output_boundary = output[:, 68:81, :, :]
    dataset='300W'
    new_out = torch.zeros((output.size(0), 68, 128, 128))
    for i in range(13):
        start_index = point_range[dataset][i][0]
        end_index = point_range[dataset][i][1]
        landmark_part = output_landmark[:, start_index:end_index, :, :]
        boundary_part = output_boundary[:, (i):(i+1), :, :]
        boundary_part = boundary_part.repeat(1, end_index-start_index, 1, 1)
        new_part = landmark_part+boundary_part
        new_out[:, start_index:end_index, :, :] = new_part
    return new_out

def get_preds3(scores, pred_pts, gt_pts): # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    #转换成numpy数据
    pred = np.zeros((1,68,2))
    scores = scores.numpy()
    dataset = '300W'
    radius=1
    for i  in range(68):
        landmark_part = scores[0, i, :, :]
        start_x = int(pred_pts[0, i, 0] - radius)
        end_x = int(pred_pts[0, i, 0] + radius)
        if start_x <= 0:
            start_x = 0
            end_x = start_x + 2*radius
        if end_x > 127:
            start_x = 127 - 2*radius
            end_x = 127
        start_y = int(pred_pts[0, i, 1] - radius)
        end_y = int(pred_pts[0, i, 1] + radius)
        if start_y <= 0:
            start_y = 0
            end_y = start_y + 2*radius
        if end_y > 127:
            start_y = 127 - 2*radius
            end_y = 127
        landmark_part_tmp = landmark_part[start_y:end_y + 1, start_x:end_x + 1]
        landmark_part_tmp = landmark_part_tmp.reshape(1, -1)
        max_val = np.argmax(landmark_part_tmp, 1) + 1
        pred[0, i, 0] = (max_val - 1) % (2*radius+1) + start_x
        pred[0, i, 1] = np.floor((max_val - 1) / (2*radius+1)) + start_y
    return pred

def get_preds4(scores, pred_pts, gt_pts): # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    #转换成numpy数据
    pred = np.zeros((1,68,2))
    scores = scores.numpy()
    dataset = '300W'
    boundary_num = 13
    radius=1
    for boundary_index in range(boundary_num):#boundary_index：0-12
        boundary_part = scores[0, 68+boundary_index, :, :]
        for kp_index in range(point_range[dataset][boundary_index][0],point_range[dataset][boundary_index][1]):
            start_x = int(pred_pts[0, kp_index, 0] - radius)
            end_x = int(pred_pts[0, kp_index, 0] + radius)
            if start_x <= 0:
                start_x = 0
                end_x = start_x + 2*radius
            if end_x > 127:
                start_x = 127 - 2*radius
                end_x = 127
            start_y = int(pred_pts[0, kp_index, 1] - radius)
            end_y = int(pred_pts[0, kp_index, 1] + radius)
            if start_y <= 0:
                start_y = 0
                end_y = start_y + 2*radius
            if end_y > 127:
                start_y = 127 - 2*radius
                end_y = 127
            boundary_part_tmp = boundary_part[start_y:end_y + 1, start_x:end_x + 1]
            boundary_part_tmp = boundary_part_tmp.reshape(1, -1)
            max_val = np.argmin(boundary_part_tmp, 1) + 1
            pred[0, kp_index, 0] = (max_val - 1) % (2*radius+1) + start_x
            pred[0, kp_index, 1] = np.floor((max_val - 1) / (2*radius+1)) + start_y
    return pred


def show_landmarks(image, landmarks):
    """Show image with landmarks"""

    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='o', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def rotatepoints(landmarks, center, rot):
    
    center_coord = np.zeros_like(landmarks)
    center_coord[:,0] = center[0]
    center_coord[:,1] = center[1]

    angle = math.radians(rot) # radians()方法将角度转换为弧度
    
    rot_matrix = np.array([[math.cos(angle), -1*math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])
    
    rotate_coords = np.dot((landmarks - center_coord) ,rot_matrix) + center_coord
    
    return rotate_coords

def enlarge_box(box, factor=0.05):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    
    xmin = xmin - width * factor
    xmax = xmax + width * factor
    ymin = ymin - height * factor
    ymax = ymax + height * factor
    
    new_box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    
    return new_box

def flip_channels(maps):
    # horizontally flip the channels
    # maps is a tensor of dimension n x c x h x w or c x h x w
    if maps.ndimension() == 4:
        maps = maps.numpy()
        maps = maps[:, :, :, ::-1].copy()
    elif maps.ndimension() == 3:
        maps = maps.numpy()
        maps = maps[:, :, ::-1].copy()
    else:
        exit('tensor dimension is not right')

    return torch.from_numpy(maps).float()

match_parts_68 = np.array([[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], # outline
            [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], # eyebrow
            [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46], # eye
            [31, 35], [32, 34], # nose
            [48, 54], [49, 53], [50, 52], [59, 55], [58, 56], # outer mouth
            [60, 64], [61, 63], [67, 65]])
match_parts_98 = np.array([[0, 32],[1, 31],[2, 30],[3,29],[4, 28],[5, 27],[6, 26],[7, 25],[8, 24],[9, 23],[10, 22],[11, 21],
                           [12, 20],[13, 19],[14, 18],[15, 17], # outline
                           [33, 46],[34, 45],[35, 44],[36, 43],[37, 42],[41, 47],[40, 48],[39, 49],[38, 50],# eyebrow
                           [60, 72],[61, 71],[62, 70],[63, 69],[64, 68],[67, 73],[66, 74],[65, 75], [96, 97], # eye
                           [55, 59],[56, 58], # nose
                           [76, 82],[77, 81],[78, 80],[87, 83],[86, 84], #outer mouth
                           [88, 92],[89, 91],[95, 93]])
def flippoints(kps, width):
    nPoints = kps.shape[0]
    assert nPoints in(68,98), 'flip {} nPoints is not supported'
    if nPoints == 98:
        pairs = match_parts_98
    else:
        pairs = match_parts_68
    fkps = kps.copy()

    for pair in pairs:
        fkps[pair[0]] = kps[pair[1]]
        fkps[pair[1]] = kps[pair[0]]
    fkps[:,0] = width - fkps[:,0] - 1
    
    return fkps

def shuffle_channels_for_horizontal_flipping(maps):
    # when the image is horizontally flipped, its corresponding groundtruth maps should be shuffled.
    # maps is a tensor of dimension n x c x h x w or c x h x w
    if maps.ndimension() == 4:
        dim = 1
        nPoints = maps.size(1)
    elif maps.ndimension() == 3:
        dim = 0
        nPoints = maps.size(0)
    else:
        exit('tensor dimension is not right')   
    if nPoints == 98:
        match_parts = match_parts_98
    else:
        match_parts = match_parts_68
    for i in range(0, match_parts.shape[0]):
        idx1, idx2 = match_parts[i]
        idx1 = int(idx1)
        idx2 = int(idx2)
        tmp = maps.narrow(dim, idx1, 1).clone() # narrow(dimension, start, length) dimension是要压缩的维度
        maps.narrow(dim, idx1, 1).copy_(maps.narrow(dim, idx2, 1))
        maps.narrow(dim, idx2, 1).copy_(tmp)
    return maps

def show_image(image, landmarks, box=None):
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    num_points = landmarks.shape[0]
    if num_points == 68:
        ax.plot(landmarks[0:17,0],landmarks[0:17,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[17:22,0],landmarks[17:22,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[22:27,0],landmarks[22:27,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[27:31,0],landmarks[27:31,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[31:36,0],landmarks[31:36,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[36:42,0],landmarks[36:42,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[42:48,0],landmarks[42:48,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[48:60,0],landmarks[48:60,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[60:68,0],landmarks[60:68,1],marker='o',markersize=4,linestyle='-',color='w',lw=2) 
    elif num_points == 98:
        ax.plot(landmarks[0:33,0],landmarks[0:33,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[33:38,0],landmarks[33:38,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[37:42,0],landmarks[37:42,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[42:46,0],landmarks[42:46,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[45:51,0],landmarks[45:51,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[51:55,0],landmarks[51:55,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[55:60,0],landmarks[55:60,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[60:65,0],landmarks[60:65,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[64:68,0],landmarks[64:68,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[68:73,0],landmarks[68:73,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[72:76,0],landmarks[72:76,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[76:83,0],landmarks[76:83,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[82:88,0],landmarks[82:88,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[88:93,0],landmarks[88:93,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[92:96,0],landmarks[92:96,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[96,0],landmarks[96,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
        ax.plot(landmarks[97,0],landmarks[97,1],marker='o',markersize=4,linestyle='-',color='w',lw=2)
    if box is not None:
        currentAxis=plt.gca()
        box = enlarge_box(box,0.05)
        xmin, ymin, xmax, ymax = box
        rect=patches.Rectangle((xmin, ymin),xmax-xmin,ymax-ymin,linewidth=2,edgecolor='r',facecolor='none')
        currentAxis.add_patch(rect)
    ax.axis('off')
    plt.show()
    
def rmse_batch_final(output, ann, tforms=None, target_type='heatmap'):
    assert target_type in ['heatmap','landmarks'], 'Only support heatmap regression and landmarks regression'
    if target_type == 'heatmap':
        output = get_new_output(output)
        pred = get_preds(output) # get_preds的返回值类型： torch.Size([12, 68, 2])
        # print('get_preds的返回值类型：',pred.size())
    else:
        new_pred = output
    pred = pred.numpy() #get_preds numpy 后的返回值类型： (12, 68, 2)
    # print('get_preds numpy 后的返回值类型：',pred.shape)
    ann = ann.numpy()
    return per_image_rmse(pred,ann,tforms)


def rmse_batch_final2(output, ann, pred_pts, tforms=None, target_type='heatmap'):
    assert target_type in ['heatmap', 'landmarks'], 'Only support heatmap regression and landmarks regression'
    if target_type == 'heatmap':
        # output = get_new_output(output)
        pred = get_preds1(output, pred_pts, ann)  # get_preds的返回值类型： torch.Size([12, 68, 2])
        # print('get_preds的返回值类型：',pred.size())
    else:
        new_pred = output
    # pred = pred.numpy()  # get_preds numpy 后的返回值类型： (12, 68, 2)
    # print('get_preds numpy 后的返回值类型：',pred.shape)
    ann = ann.numpy()

    return per_image_rmse(pred,ann,tforms)

def rmse_batch_final3(output, ann, pred_pts, tforms=None, target_type='heatmap'):
    assert target_type in ['heatmap', 'landmarks'], 'Only support heatmap regression and landmarks regression'
    if target_type == 'heatmap':
        output = get_new_output(output)
        pred = get_preds2(output, pred_pts, ann)  # get_preds的返回值类型： torch.Size([12, 68, 2])
        # print('get_preds的返回值类型：',pred.size())
    else:
        new_pred = output
    # pred = pred.numpy()  # get_preds numpy 后的返回值类型： (12, 68, 2)
    # print('get_preds numpy 后的返回值类型：',pred.shape)
    ann = ann.numpy()

    return per_image_rmse(pred,ann,tforms)

def rmse_batch_final4(output, ann, pred_pts, tforms=None, target_type='heatmap'):
    assert target_type in ['heatmap', 'landmarks'], 'Only support heatmap regression and landmarks regression'
    if target_type == 'heatmap':
        output = get_new_output(output)
        pred = get_preds2(output, pred_pts, ann)  # get_preds的返回值类型： torch.Size([12, 68, 2])
        # print('get_preds的返回值类型：',pred.size())
    else:
        new_pred = output
    # pred = pred.numpy()  # get_preds numpy 后的返回值类型： (12, 68, 2)
    # print('get_preds numpy 后的返回值类型：',pred.shape)
    ann = ann.numpy()

    return per_image_rmse(pred,ann,tforms)

def rmse_batch(output, ann, tforms=None, target_type='heatmap'):
    assert target_type in ['heatmap', 'landmarks'], 'Only support heatmap regression and landmarks regression'
    if target_type == 'heatmap':
        pred = get_preds(output[:, 0:68, :, :])  # get_preds的返回值类型： torch.Size([12, 68, 2])
        # print('get_preds的返回值类型：',pred.size())
    else:
        new_pred = output
    pred = pred.numpy()  # get_preds numpy 后的返回值类型： (12, 68, 2)
    # print('get_preds numpy 后的返回值类型：',pred.shape)
    ann = ann.numpy()
    return per_image_rmse(pred, ann, tforms), pred


def get_subpixel_from_kpts(kpts, heatmaps, cut_szie):
    """

    Args:
        heatmaps: [batch, size, size, num_lmk]
        kpts:[batch, num_lmk, 2]
    Returns:

    """
    n, num_lmk, h, w = heatmaps.shape
    kpt_s = np.zeros((num_lmk,2))
    for land_index in range(num_lmk):
        kpt = kpts[0,land_index,:] #取第一个特征点的坐标
        hm = heatmaps[0, land_index, :, :] #取第一个特征点的热力图
        kpt_x = int(kpt[0])
        kpt_y = int(kpt[1])
        if kpt_x-cut_szie<0:
            left_x = 0
            right_x = 2*cut_szie+1
        elif kpt_x+cut_szie>127:
            left_x = 127-(2*cut_szie+1)
            right_x = 127
        else:
            left_x = kpt_x-cut_szie
            right_x = kpt_x+cut_szie

        if kpt_y-cut_szie<0:
            bottom_y = 0
            top_y = 2*cut_szie+1
        elif kpt_y+cut_szie>127:
            bottom_y = 127-(2*cut_szie+1)
            top_y = 127
        else:
            bottom_y = kpt_y-cut_szie
            top_y = kpt_y+cut_szie

        hm_part = hm[bottom_y:top_y+1, left_x:right_x+1]
        m_iN = (2*cut_szie+1) * (2*cut_szie+1)
        iPos = 0
        tmp_a = np.zeros((m_iN,1))
        tmp_b = np.zeros((m_iN,5))
        for ii in range(2*cut_szie+1):
            for jj in range(2*cut_szie+1):
                tmp = hm_part[ii,jj]
                tmp_a[iPos,0] = tmp*np.log(float(tmp))
                tmp_b[iPos,0] = tmp
                tmp_b[iPos, 1] = tmp*ii
                tmp_b[iPos, 2] = tmp*jj
                tmp_b[iPos, 3] = tmp*ii*ii
                tmp_b[iPos, 4] = tmp*jj*jj
                iPos = iPos + 1
        qq, rr = np.linalg.qr(tmp_b,mode='complete')
        ss =np.dot(qq.T,tmp_a)
        ss = ss[0:5]
        rr1 = rr[0:5, 0:5]
        cc = np.dot(np.linalg.inv(rr1),ss)
        kpt_x = left_x - 0.5 * cc[2] / cc[4]
        kpt_y = bottom_y - 0.5 * cc[1] / cc[3]
        kpt_s[land_index,0] = kpt_x
        kpt_s[land_index, 1] = kpt_y
    return kpt_s




def get_new_plus_output(output): # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert output.dim() == 4, 'Score maps should be 4-dim'
    output_landmark = output[:, 0:68, :, :]
    output_boundary = output[:, 68:81, :, :]
    dataset='300W'
    new_out = torch.zeros((output.size(0), 68, 128, 128))
    for i in range(13):
        start_index = point_range[dataset][i][0]
        end_index = point_range[dataset][i][1]
        landmark_part = output_landmark[:, start_index:end_index, :, :]
        boundary_part = output_boundary[:, (i):(i+1), :, :]
        boundary_part = boundary_part.repeat(1, end_index-start_index, 1, 1)
        new_part = landmark_part+boundary_part
        new_out[:, start_index:end_index, :, :] = new_part
    return new_out


def rmse_batch_final5(output, ann, pred_pts, tforms=None, target_type='heatmap'):
    assert target_type in ['heatmap', 'landmarks'], 'Only support heatmap regression and landmarks regression'
    if target_type == 'heatmap':
        output1 = get_new_output(output) #相乘
        pred1 = get_preds3(output1, pred_pts, ann)
        pred2 = get_preds4(output, pred_pts, ann) #边界
        output3 = get_new_plus_output(output) #相加
        pred3 = get_preds3(output3, pred_pts, ann)
        pred = pred2
        # get_preds的返回值类型： torch.Size([12, 68, 2])
        # print('get_preds的返回值类型：',pred.size())
    else:
        new_pred = output
    # pred = pred.numpy()  # get_preds numpy 后的返回值类型： (12, 68, 2)
    # print('get_preds numpy 后的返回值类型：',pred.shape)
    ann = ann.numpy()

    return per_image_rmse(pred,ann,tforms)
    
def per_image_rmse(pred, ann, tform=None):
    # pred: N x L x 2 numpy
    # ann:  N x L x 2 numpy
    # rmse: N numpy 

    N = pred.shape[0]
    L = pred.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = pred[i], ann[i]
    
        # project to origin resolution ！！！！！！！！！！！！
        # linalg=linear（线性）+algebra（代数），norm则表示范数，ord=2默认二范数
        pts_pred = np.dot(pts_pred-tform['translation'][i].numpy(),np.linalg.inv(tform['rotation'][i].numpy() * tform['scale'][i].numpy()))
        pts_gt = np.dot(pts_gt-tform['translation'][i].numpy(),np.linalg.inv(tform['rotation'][i].numpy() * tform['scale'][i].numpy()))
        if L == 98:
            interpupil = np.linalg.norm(pts_gt[96,:]-pts_gt[97,:])
        elif L == 68:
            #interocular
            interocular = np.linalg.norm(pts_gt[36,:]-pts_gt[45,:])
            # interpupil
            # lcenter = (pts_gt[36,:]+pts_gt[37,:]+pts_gt[38,:]+pts_gt[39,:]+pts_gt[40,:]+pts_gt[41,:])/6
            # rcenter = (pts_gt[42,:]+pts_gt[43,:]+pts_gt[44,:]+pts_gt[45,:]+pts_gt[46,:]+pts_gt[47,:])/6
            # interpupil = np.linalg.norm(lcenter-rcenter)
        rmse[i] = np.sum(np.linalg.norm(pts_pred-pts_gt, axis=1))/(interocular*L)
    return rmse

def _plot_curves(bins, ced_values, legend_entries, title, x_limit=0.08,
                 colors=None, linewidth=3, fontsize=12, figure_size=(11,6)):
    # number of curves
    n_curves = len(ced_values)

    # if no colors are provided, sample them from the jet colormap
    if colors is None:
        cm = plt.get_cmap('jet')
        colors = [cm(1.*i/n_curves)[:3] for i in range(n_curves)]
        
    # plot all curves
    fig = plt.figure()
    ax = plt.gca()
    for i, y in enumerate(ced_values):
        plt.plot(bins, y, color=colors[i],
                 linestyle='-',
                 linewidth=linewidth, 
                 label=legend_entries[i])
        
    # legend
    ax.legend(prop={'size': fontsize}, loc=0)
    
    # axes
    for l in (ax.get_xticklabels() + ax.get_yticklabels()):
        l.set_fontsize(fontsize)
    ax.set_xlabel('Point-to-point Normalized RMS Error', fontsize=fontsize)
    ax.set_ylabel('Images Proportion', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # set axes limits
    ax.set_xlim([0., x_limit])
    ax.set_ylim([0., 1.])
    ax.set_yticks(np.arange(0., 1.1, 0.1))
    
    # grid
    plt.grid('on', linestyle='--', linewidth=0.5)
    
    # figure size
    if figure_size is not None:
        fig.set_size_inches(np.asarray(figure_size))
        
def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def gaussian(shape=(7,7),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return torch.from_numpy(h).float()

def color_heatmap(x):
    x = x.numpy()
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def draw_gaussian(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian 
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    

    # Check that any part of the gaussian is in-bounds
    img = np.ones(( 128, 128))/100000
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return torch.from_numpy(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform
 