import torch
import torch.nn as nn
from model.YOLO3D import YOLO3D
from util.calculate import *

class Detection_Framework(nn.Module):
    """ Full frame of YOLO3D.
        All data needed are calculated here.

        Only return PREDICTION and LOSS for detection or back propagation.
    """

    def __init__(self, in_channels=3, class_num=1, anchor_num=3, anchor=None, calculate_loss=True):
        """
            anchor: 3-dims tensor [3, anchor_num, 1], like below.
            [[[5], [10], [15]], 
             [[20], [40], [60]], 
             [[80], [100], [120]]]
        """

        super(Detection_Framework, self).__init__()
        self.class_num = class_num
        self.anchor_num = anchor_num
        self.anchor = anchor
        self.network = YOLO3D(in_channels=in_channels, out_channels=(5 + self.class_num) * self.anchor_num)
        self.calculate_loss = calculate_loss 

        ########## parameters to set
        self.ignore_thres = 0.333
        self.obj_scale = 1
        self.noobj_scale = 0.1
        self.bbox_scale = 5
        self.class_scale = 5
        ##########

        #self.mse_loss = nn.MSELoss(reduction='sum')
        self.mse_loss = nn.SmoothL1Loss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
    
    def forward(self, images, targets):
        ByteTensor = torch.cuda.ByteTensor if images.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if images.is_cuda else torch.FloatTensor

        layer_out1, layer_out2, layer_out3 = self.network(images)

        ANCHOR = self.anchor.type(FloatTensor)

        """
        Change
        """
        #pred1, cal1, sa1 = self.make_prediction(layer_out1, ANCHOR[2, ...], 16)#return size, like [batch, detect_num, 85]
        #pred2, cal2, sa2 = self.make_prediction(layer_out2, ANCHOR[1, ...], 8)
        #pred3, cal3, sa3 = self.make_prediction(layer_out3, ANCHOR[0, ...], 4)

        #pred = torch.cat((pred1, pred2, pred3), dim=1)
        #cal = torch.cat((cal1, cal2, cal3), dim=1)#participates in back propagation
        #sa = torch.cat((sa1, sa2, sa3), dim=0)
        pred, cal, sa = self.make_prediction(layer_out3, ANCHOR[0, ...], 4)
        
        if not self.calculate_loss:#for test
            return pred, 0, 0, 0
        else:#for training and eval
            """ Generate predictions and targets for loss:
                    Selects target check boxes for bounding box loss and class loss.
                    Prepares conference loss.
                    Ignores non-target check boxes that have large iou with ground truth boxes
            """
            batch_size = pred.size(0)
            detect_num = pred.size(1)#total number of check boxes, fo 416x416 image is 10647

            pred_bbox = FloatTensor(targets.size()).fill_(0)#size [batch, detect_num, 85]
            pred_conf = cal[..., 4]

            target_bbox = FloatTensor(targets.size()).fill_(0)
            bbox_mask = targets[..., 4] > 0.5#those bounding boxes with objects
            target_conf = FloatTensor(batch_size, detect_num).fill_(0)

            #only for calculating conference loss
            obj_mask = ByteTensor(batch_size, detect_num).fill_(0)
            bg_mask = ByteTensor(batch_size, detect_num).fill_(1)

            #prepare data above for calculating loss
            #left for generate: pred_bbox, target_bbox, target_conf
            for b in range(batch_size):
                mask = bbox_mask[b]#mask of objects in this batch
                target = targets[b, mask]#target boxes in this batch

                if target.size(0) == 0:#no objects
                    continue

                """ Use much space! 
                    Make sure targets is not too many, or set a limit for targets
                """
                ious = cross_calculate_iou(target[:, :4], pred[b, :, :4])

                """ Little Bug:
                        Two ground truth objects may have same corresponding checking boxes.
                """
                _, max_index = ious.max(1)#index of check boxes which have the best iou with one of those bounding boxes in target

                obj_mask[b, max_index] = 1
                bg_mask[b, max_index] = 0
                target_conf[b, max_index] = 1
                pred_bbox[b, mask, :] = cal[b, max_index, :]

                target_bbox[b, mask, 0] = (target[:, 0] / sa[max_index, 0]).frac()
                target_bbox[b, mask, 1] = (target[:, 1] / sa[max_index, 0]).frac()
                target_bbox[b, mask, 2] = (target[:, 2] / sa[max_index, 0]).frac()
                target_bbox[b, mask, 3] = torch.log(target[:, 3] / sa[max_index, 1])
                target_bbox[b, mask, 4:] = target[:, 4:]

                ignore_mask = torch.sum(ious > self.ignore_thres, 0) >= 1#mask for those check boxes to be thrown
                bg_mask[b, ignore_mask] = 0
            
            """ Calculates loss
                mse_loss for bounding box loss
                bce_loss for conference loss and class loss
            """
            obj_mask = obj_mask.bool()
            bg_mask = bg_mask.bool()

            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], target_conf[obj_mask]) if obj_mask.any() else FloatTensor([0])
            loss_conf_bg = self.bce_loss(pred_conf[bg_mask], target_conf[bg_mask]) if bg_mask.any() else FloatTensor([0])

            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_bg

            pred_x = pred_bbox[..., 0]
            pred_y = pred_bbox[..., 1]
            pred_z = pred_bbox[..., 2]
            pred_r = pred_bbox[..., 3]
            pred_class = pred_bbox[..., 5:]

            target_x = target_bbox[..., 0]
            target_y = target_bbox[..., 1]
            target_z = target_bbox[..., 2]
            target_r = target_bbox[..., 3]
            target_class = target_bbox[..., 5:]

            loss_x = self.mse_loss(pred_x[bbox_mask], target_x[bbox_mask]) if bbox_mask.any() else FloatTensor([0])
            loss_y = self.mse_loss(pred_y[bbox_mask], target_y[bbox_mask]) if bbox_mask.any() else FloatTensor([0])
            loss_z = self.mse_loss(pred_z[bbox_mask], target_z[bbox_mask]) if bbox_mask.any() else FloatTensor([0])
            loss_r = self.mse_loss(pred_r[bbox_mask], target_r[bbox_mask]) if bbox_mask.any() else FloatTensor([0])

            loss_bbox = (loss_x + loss_y + loss_z + loss_r) * self.bbox_scale
            loss_cls = (self.bce_loss(pred_class[bbox_mask], target_class[bbox_mask]) if bbox_mask.any() else FloatTensor([0])) * self.class_scale
            
            loss_bbox = loss_bbox / batch_size
            loss_conf = loss_conf / batch_size
            loss_cls = loss_cls / batch_size
            
            total_loss = loss_bbox + loss_conf + loss_cls
            return pred, total_loss.view(1, 1), loss_conf.view(1, 1), loss_cls.view(1, 1)
    
    
    def make_prediction(self, output, anchor, scale):
        """ Make YOLO bbox from network output.
            For 3D detection.
            
            Inputs:
                output: yolo network output
                anchor: anchors of this detection layer
                scale: raw image's size / this layer's size, in this inplement are 8, 16, 32
                
            Returns:
                pred: prediction at origin scale of the images for detection
                cal: prediction that participates in calculating loss and back propagation
                sa: scale and anchors for calculating loss
        """
        FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor
        batch_size = output.size(0)
        gx = output.size(-1)
        gy = output.size(-2)
        gz = output.size(-3)

        tmp = output.view(batch_size, self.anchor_num, -1, gz, gy, gx).permute(0, 1, 3, 4, 5, 2).contiguous()

        x = torch.sigmoid(tmp[..., 0])
        y = torch.sigmoid(tmp[..., 1])
        z = torch.sigmoid(tmp[..., 2])
        r = tmp[..., 3]
        conf = torch.sigmoid(tmp[..., 4])
        classes = nn.functional.softmax(tmp[..., 5:], dim=-1)

        grid_x = torch.arange(gx).repeat(gz, gy, 1).view(1, 1, gz, gy, gx).type(FloatTensor)
        grid_y = torch.arange(gy).repeat(gx, 1).t().repeat(gz, 1, 1).view(1, 1, gz, gy, gx).type(FloatTensor)
        grid_z = torch.arange(gz).repeat(gx * gy, 1).t().view(1, 1, gz, gy, gx).type(FloatTensor)

        anchor_r = anchor.view(1, self.anchor_num, 1, 1, 1)

        pred_boxes = FloatTensor(tmp[..., :4].shape)
        pred_boxes[..., 0] = (x + grid_x) * scale
        pred_boxes[..., 1] = (y + grid_y) * scale
        pred_boxes[..., 2] = (z + grid_z) * scale
        pred_boxes[..., 3] = torch.exp(r) * anchor_r

        pred = torch.cat((pred_boxes.view(batch_size, -1, 4),
                          conf.view(batch_size, -1, 1),
                          classes.view(batch_size, -1, self.class_num)), dim=-1).detach()
        
        cal = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1), r.unsqueeze(-1), conf.unsqueeze(-1), classes), dim=-1).view(batch_size, -1, self.class_num + 5)

        #scale and anchor used for calculating loss later
        detect_num =  pred.size(1)
        sa = FloatTensor(detect_num, 2)
        sa[..., 0] = scale
        for i in range(self.anchor_num):
            sa[gx * gy * gz * i: gx * gy * gz * (i + 1), 1:] = anchor[i].repeat(gx * gy * gz, 1)
        
        return pred, cal, sa
    
    def setCalculateLoss(self, state=True):
        self.calculate_loss = state