""" Functions for calculating.
    For 3D detection model.
    All data here are torch.tensor (more specifically, torch.cuda.tensor).
"""

import torch

def cross_calculate_iou(box1, box2):
    """
        Calculate cross iou of two bounding box groups.
        Returns cross map with size [len(box1), len(box2)].
    """
    len1 = box1.size(0)
    len2 = box2.size(0)
    #repeat to same size and calculate
    BOX1 = box1.view(-1, 1, 4).repeat(1, len2, 1).view(-1, 4)
    BOX2 = box2.repeat(len1, 1)

    iou = calculate_iou(BOX1, BOX2)
    iou = iou.view(len1, len2).contiguous()

    return iou

def calculate_iou(box1, box2):
    """
        Calculate iou of two boxes tensor.
        Make sure all data > 0.
        Returns the IoU of two bounding boxes.
    """
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 3] / 2, box1[:, 0] + box1[:, 3] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b1_z1, b1_z2 = box1[:, 2] - box1[:, 3] / 2, box1[:, 2] + box1[:, 3] / 2

    b2_x1, b2_x2 = box2[:, 0] - box2[:, 3] / 2, box2[:, 0] + box2[:, 3] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    b2_z1, b2_z2 = box2[:, 2] - box2[:, 3] / 2, box2[:, 2] + box2[:, 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_z1 = torch.max(b1_z1, b2_z1)

    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_rect_z2 = torch.min(b1_z2, b2_z2)

    inter_vol = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0) * torch.clamp(inter_rect_z2 - inter_rect_z1, min=0)

    b1_vol = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) * (b1_z2 - b1_z1)
    b2_vol = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) * (b2_z2 - b2_z1)

    iou = inter_vol / (b1_vol + b2_vol - inter_vol + 1e-6)
    return iou

# For predction
def Non_Maximum_Suppression(boxes, nms_thres=0.3):
    """ Non maximum suppression
        Removes unnecessary check boxes
        Input:
            boxes: tensor of bounding boxes, size [num, 5], left out class for space
            nms_thres: threshold of iou
        
        Returns:
            left_index: index of chosen bounding boxes with one dimension
    """
    conf_slice = boxes[:, 4]
    _, index = torch.sort(conf_slice, descending=True)
    descending_boxes = boxes[index, :]

    iou = cross_calculate_iou(descending_boxes[:, :4], descending_boxes[:, :4])

    for i in range(descending_boxes.size(0)):
        if descending_boxes[i, 4] > 1e-6:
            remove_mask = iou[i, :] > nms_thres
            remove_mask[i] = False
            descending_boxes[remove_mask, 4] = 0
    
    left_index = index[descending_boxes[:, 4] > 1e-6]

    return left_index

def remove(boxes):
    """ Remove close candidates. 
        If one candidate's location is in the detection area of another candidate with higher confidence, remove it. 
        We choose 2 / 3 of prediction radius as the occupied area of one candidates. 

        Input:
            boxes: tensor of candidates, size [num, 5], left out class for space
        
        Returns:
            left_index: index of chosen candidates with one dimension
    """
    conf_slice = boxes[:, 4]
    _, index = torch.sort(conf_slice, descending=True)
    descending_boxes = boxes[index, :]

    iou = cross_calculate_des(descending_boxes[:, :4], descending_boxes[:, :4])

    for i in range(descending_boxes.size(0)):
        if descending_boxes[i, 4] > 1e-6:
            remove_mask = iou[i, :] <= descending_boxes[i, 3] * 2 / 3
            remove_mask[i] = False
            descending_boxes[remove_mask, 4] = 0
    
    left_index = index[descending_boxes[:, 4] > 1e-6]
    #print('left_index: ', len(left_index))
    return left_index

def cross_calculate_des(box1, box2):
    """
        Calculate cross distance of two bounding box groups.
        Returns cross map with size [len(box1), len(box2)].
    """
    len1 = box1.size(0)
    len2 = box2.size(0)
    #repeat to same size and calculate
    BOX1 = box1.view(-1, 1, 4).repeat(1, len2, 1).view(-1, 4)
    BOX2 = box2.repeat(len1, 1)

    des = calculate_des(BOX1, BOX2)
    des = des.view(len1, len2).contiguous()

    return des

def calculate_des(box1, box2):
    """
        Calculate distance of two boxes tensor.
        Make sure all data > 0.
        Returns the distance of two bounding boxes.
    """
    return torch.sqrt(torch.pow(box2[:, 0] - box1[:, 0], 2) + torch.pow(box2[:, 1] - box1[:, 1], 2) + torch.pow(box2[:, 2] - box1[:, 2], 2))