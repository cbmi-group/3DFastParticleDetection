import os
import mrcfile
import warnings
import numpy as np
import pandas as pd
from skimage.morphology import dilation

from .calculate import calculate_iou
import torch

def eval(pred_list, gt_list, loc2id, class_id=0):
    """ Evaluation without class(one class).
        All data here is ndarray.

        Input:
            pred_list: List of particles predicted by network. Like: [[x, y, z]]
            gt_list: Ground truth location.
            loc2id: Cube to relate predicted location and ground truth point id.
            class_id: 1-12 for one class evaluation. 0 for all. 
    """
    if class_id > 0:
        pred_list_of_one_class = pred_list[pred_list[:, 5] == class_id]
    else:
        pred_list_of_one_class = pred_list
    
    pred_x = pred_list_of_one_class[:, 0]
    pred_y = pred_list_of_one_class[:, 1]
    pred_z = pred_list_of_one_class[:, 2]

    result = loc2id[pred_z, pred_y, pred_x].astype(np.int64)
    gt_class = gt_list[result][:, 0]

    """
    #Average distance
    count = 0
    dis_sum = 0
    for i in range(len(pred_list_of_one_class)):
        if class_id == 0 and gt_list[i, 0] != 0 or gt_list[i, 0] == class_id:
            count += 1
            dis_sum += np.sqrt(np.sum(np.square(pred_list_of_one_class[i, :3] - gt_list[result[i], 1:4])))
    print(dis_sum / count)
    """

    if class_id > 0:
        gt_box = ((gt_list[:, 0] == class_id).astype(np.int16) - 1) * 2#positive box with 0 and negative box with -2
    else:
        gt_box = np.zeros(gt_list.shape[0])
    gt_box[result] += 1

    FP = sum(gt_class == 0) if class_id == 0 else sum(gt_class != class_id)
    FN = sum(gt_box[1:] == 0)
    TP = sum(gt_box[1:] > 0)
    dp = pred_list_of_one_class.shape[0] - FP - TP#duplicate
    
    return TP, FP, FN, dp

def multi_class_eval(pred_list, gt_list, loc2id):
    """ Multi-Class Evaluation.
        All data here is ndarray.

        Input:
            pred_list: List of particles predicted by network. Like: [[x, y, z]]
            gt_list: Ground truth location.
            loc2id: Cube to relate predicted location and ground truth point id.
    """
    PRECISION = np.zeros(12, dtype=np.float32)
    RECALL = np.zeros(12, dtype=np.float32)
    for i in range(12):
        TP, FP, FN, _ = eval(pred_list, gt_list, loc2id, class_id=i + 1)
        PRECISION[i] = TP / (TP + FP + 1e-6)
        RECALL[i] = TP / (TP + FN + 1e-6)
    
    pred_x = pred_list[:, 0]
    pred_y = pred_list[:, 1]
    pred_z = pred_list[:, 2]
    pred_class = pred_list[:, 5]
    result = loc2id[pred_z, pred_y, pred_x].astype(np.int64)
    gt_class = gt_list[result][:, 0]

    result_map = np.zeros((12, 13), dtype=np.int64)
    for i in range(len(result)):
        result_map[pred_class[i] - 1, gt_class[i]] += 1
    print(result_map)

    """
    #old codes, without eliminating multi hits

    gt_calculator = np.zeros(13, dtype=np.int64)
    for i in range(len(gt_list)):
        gt_calculator[gt_list[i, 0]] += 1
    #precision and recall
    precision = np.zeros(12, dtype=np.float32)
    recall = np.zeros(12, dtype=np.float32)
    for i in range(12):
        l = result_map[i]
        precision[i] = l[i + 1] / (l.sum() + 1e-6)
        recall[i] = l[i + 1] / (gt_calculator[i + 1] + 1e-6)
    
    print('precision: ', precision)
    print('recall: ', recall)
    print('F1 score: ', 2 * precision * recall / (precision + recall + 1e-6))
    """

    print('precision: ', PRECISION)
    print('recall: ', RECALL)
    print('F1 score: ', 2 * PRECISION * RECALL / (PRECISION + RECALL + 1e-6))

    """
    #calculate mean IOU
    gt_box = gt_list[result]
    pred_box = pred_list[:, 0:4]
    
    pred_box = pred_box[gt_box[:, 0] > 0, :]
    gt_box = gt_box[gt_box[:, 0] > 0, :]

    tv = np.array([0, 13, 6, 12, 16, 10, 9, 10, 6, 7, 12, 12, 11], dtype=np.int16)
    gt_box = gt_box.astype(np.int32)
    gt_box[:, 0] = tv[gt_box[:, 0]]
    gt_box = torch.from_numpy(gt_box)
    pred_box = torch.from_numpy(pred_box)
    gt_box = torch.cat((gt_box[:, 1:4], gt_box[:, 0:1]), dim=1)
    iou = calculate_iou(pred_box, gt_box)

    print("mean IOU: ", iou.mean())
    """

def SHREC2020_EVAL(pred_list, base_dir='/ldap_shared/synology_shared/em_data/ET/shrec_2020/shrec2020_full/', data_id=8, multi_class=True):
    """
        data_id: 8 for eval and 9 for test. 
    """
    location = pd.read_csv(os.path.join(base_dir, 'model_%d/particle_locations.txt' % data_id), header=None, sep=' ')
    particle_dict = {'3cf3': 1, '1s3x': 2, '1u6g': 3, '4cr2': 4, '1qvr': 5, '3h84': 6, '2cg9': 7, '3qm1': 8, '3gl1': 9, '3d2f': 10, '4d8q': 11, '1bxn': 12}
    for i in range(len(location)):
        location.loc[i, 0] = particle_dict[location.loc[i, 0]]
    particle_list = np.array(location.loc[:, 0:3])
    gt_list = np.concatenate((np.array([[0, 0, 0, 0]]), particle_list), axis=0)

    warnings.simplefilter('ignore')
    occupancy_map = None
    with mrcfile.open(os.path.join(base_dir, 'model_%d/occupancy_mask.mrc' % data_id), permissive=True) as m:
        occupancy_map = dilation(m.data)
    
    if multi_class:
        multi_class_eval(pred_list, gt_list, occupancy_map)
    
    TP, FP, FN, DP = eval(pred_list, gt_list, occupancy_map)
    print('TP: %d, FP: %d, FN: %d, DP: %d \t Precision: %.6f, Recall: %.6f\n' % (TP, FP, FN, DP, TP / (TP + FP + 1e-6), TP / (TP + FN + 1e-6)))

#no maintainance
def Dataset10045_EVAL(pred_list, base_dir='/ldap_shared/synology_shared/10045/10045_icon_fit/bin4_iconmask2/', data_id=10):
    """ Just one class
    """
    location = pd.read_csv(os.path.join(base_dir, 'coords/IS002_291013_0%02d_iconmask2_norm_rot_cutZ.coords' % data_id), sep='\t', header=None)
    particle_list = np.array(location)
    gt_list = np.concatenate((np.array([[0, 0, 0]]), particle_list), axis=0)

    occupancy_map = None
    with mrcfile.open(os.path.join(base_dir, 'data_ocp23/IS002_291013_0%02d_iconmask2_norm_rot_cutZ.mrc' % data_id), permissive=True) as m:
        occupancy_map = dilation(m.data)
    
    TP, FP, FN, DP = eval(pred_list, gt_list, occupancy_map)
    print('TP: %d, FP: %d, FN: %d, DP: %d \t Precision: %.6f, Recall: %.6f\n' % (TP, FP, FN, DP, TP / (TP + FP + 1e-6), TP / (TP + FN + 1e-6)))


