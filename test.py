import argparse
import torch
import numpy as np
from tqdm import tqdm

from util.utils import *
from util.eval import SHREC2020_EVAL
from dataset.SHREC3D import SHREC3D
from Detection_Framework import Detection_Framework
from util.calculate import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #project options
    parser.add_argument('--model_name', type=str, default='YOLOv3', help='Name of this experiment.')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='GPU ids, use -1 for CPU.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Models are saved here.')
    parser.add_argument('--load_dir', type=str, default='./checkpoints', help='The directory of the pretrained model.')
    """important"""
    parser.add_argument('--dataset_dir', type=str, default='/ldap_shared/synology_shared/em_data/ET/shrec_2020/shrec2020_full/', help='The directory of the used dataset')

    #testing options
    parser.add_argument('--load_filename', type=str, default='YOLOv3_EPOCH[80].pth', help='Filename of the pretrained model.')

    #dataset options
    parser.add_argument("--batch_size", type=int, default=32, help="Size of each image batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    ##########preparing
    device = prepare_devices(opt)

    eval_dataset = SHREC3D(mode='test', base_dir=opt.dataset_dir)
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
    ANCHOR = torch.tensor([[[5], [10], [15]]])
    ##########
    
    model = Detection_Framework(in_channels=1, class_num=12, anchor=ANCHOR)
    model = load_model(model, opt)

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

        total_pred_list = result[0]
        final_pred_list = total_pred_list[remove(torch.from_numpy(total_pred_list[:, :5].astype(np.float32))), :]
        
        print('EVAL Result:')
        print('Loss: %.6f \t Conf Loss: %.6f \t Class Loss: %.6f' % (np.mean(loss_list), np.mean(loss_conf_list), np.mean(loss_cls_list)))
        SHREC2020_EVAL(final_pred_list, base_dir=opt.dataset_dir, data_id=9)
        