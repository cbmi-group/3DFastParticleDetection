import argparse
import torch
import numpy as np
from tqdm import tqdm

from util.utils import *
from util.eval import Dataset10045_EVAL
from dataset.Dataset_10045 import Dataset_10045
from Detection_Framework import Detection_Framework
from util.calculate import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #project options
    parser.add_argument('--model_name', type=str, default='YOLOv3', help='Name of this experiment.')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='GPU ids, use -1 for CPU.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Models are saved here.')
    parser.add_argument('--load_dir', type=str, default='./checkpoints', help='The directory of the pretrained model.')

    #training options
    parser.add_argument('--total_epoches', type=int, default=300, help='Total epoches.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Interval between saving model weights')
    parser.add_argument('--evaluation_interval', type=int, default=10, help='Interval between evaluations on validation set')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained model.')
    parser.add_argument('--load_filename', type=str, default='YOLOv3_EPOCH[200].pth', help='Filename of the pretrained model.')

    #dataset options
    parser.add_argument("--batch_size", type=int, default=32, help="Size of each image batch.")
    #parser.add_argument("--img_size", type=int, default=416, help="Size of each image.")
    parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    ##########preparing
    device = prepare_devices(opt)

    train_dataset = Dataset_10045(mode='train')
    train_data = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=opt.num_workers, 
        collate_fn=train_dataset.collate_fn
    )

    eval_dataset = Dataset_10045(mode='val')
    eval_data = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        collate_fn=eval_dataset.collate_fn
    )

    """
        Pre-cluster results, 9 types
    """
    ##########
    ANCHOR = torch.tensor([[[24]]])
    ##########
    
    model = Detection_Framework(in_channels=1, anchor_num=1, anchor=ANCHOR)
    if opt.pretrained:
        model = load_model(model, opt)
    else:#training from begining
        if opt.gpu_num > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
    
    
    ##########
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)
    ##########

    for epoch in range(1, opt.total_epoches + 1):
        print('**********')
        print('Training Epoch %d' % epoch)
        print('Learning rate: %.6f' % scheduler.get_last_lr()[0])
        model.train()
        loss_list = []
        loss_conf_list = []
        loss_cls_list = []
        for batch_i, (imgs, targets) in enumerate(tqdm(train_data, desc=f"Epoch {epoch}")):
            imgs = imgs.to(device)
            targets = targets.to(device)
            _, loss, loss_conf, loss_cls = model(imgs, targets)
            loss = loss.mean()
            loss_conf = loss_conf.mean()
            loss_cls = loss_cls.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.cpu().detach())
            loss_conf_list.append(loss_conf.cpu().detach())
            loss_cls_list.append(loss_cls.cpu().detach())

            # ------------
            # Log pregress
            # ------------
        print('End of training epoch %d / %d \t Loss: %.6f \t Conf Loss: %.6f \t Class Loss: %.6f \n' % (epoch, opt.total_epoches, np.mean(loss_list), np.mean(loss_conf_list), np.mean(loss_cls_list)))
        
        #evaluation step
        if epoch % opt.evaluation_interval == 0:
            print("---- Evaluating Model ----")
            model.eval()

            ##########
            conf_thres = 0.5
            nms_thres = 0.1
            ##########

            loss_list = []
            loss_conf_list = []
            loss_cls_list = []

            with torch.no_grad():
                pred_list = []
                for batch_i, (imgs, targets) in enumerate(tqdm(eval_data, desc=f"Calculating")):
                    imgs = imgs.to(device)
                    device_targets = targets.to(device)
                    pred, loss, loss_conf, loss_cls = model(imgs, device_targets)
                    loss = loss.mean()
                    loss_conf = loss_conf.mean()
                    loss_cls = loss_cls.mean()
                    loss_list.append(loss.cpu().detach())
                    loss_conf_list.append(loss_conf.cpu().detach())
                    loss_cls_list.append(loss_cls.cpu().detach())

                    for b in range(pred.size(0)):
                        b_pred = pred[b, pred[b, :, 4] > conf_thres, :]
                        b_pred = b_pred.cpu()
                        index = Non_Maximum_Suppression(b_pred[:, :5], nms_thres)
                        b_pred = b_pred[index, :]
                        pred_list.append(b_pred)
                
                result = eval_dataset.joint(pred_list)
                TP, FP, FN, DP = Dataset10045_EVAL(result[0])

                
                print('EVAL Result:')
                print('Loss: %.6f \t Conf Loss: %.6f \t Class Loss: %.6f' % (np.mean(loss_list), np.mean(loss_conf_list), np.mean(loss_cls_list)))
                print('TP: %d, FP: %d, FN: %d, DP: %d \t Precision: %.6f, Recall: %.6f\n' % (TP, FP, FN, DP, TP / (TP + FP + 1e-6), TP / (TP + FN + 1e-6)))
                    
        
        if epoch %opt.checkpoint_interval == 0:
            save_model(model, opt, epoch)
        
        scheduler.step()
        