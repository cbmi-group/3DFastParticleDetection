""" Functions for devices checking, model saving and loading.
"""

import os
import torch
import torch.nn as nn

#from collections import OrderedDict

def prepare_devices(opt):
    """ Prepare device for training or testing.
        Using cpu is not recommended.
    """
    if opt.gpu_ids != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        if torch.cuda.is_available():
            opt.gpu_num = torch.cuda.device_count()#used by function below
            device = torch.device("cuda:0")
            print('Using %s GPU(s) for this project.' % torch.cuda.device_count())
        else:
            device = torch.device("cpu")
            opt.gpu_num = 0
            print('CUDA is unavailable, using CPU only.')
    else:
        device = torch.device("cpu")
        opt.gpu_num = 0
        print('Using CPU only.')
    
    opt.device = device
    return device



def save_model(model, opt, epoch):
    """ Save the patameters of the model.
        Always save model without "module" (just on one device).
    """
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    save_filename = '%s_EPOCH[%s].pth' % (opt.model_name, epoch)
    save_path = os.path.join(opt.save_dir, save_filename)
    if opt.gpu_num > 1:
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)



def load_model(model, opt):
    """ Load the parameters of the model.
        DO EVERYTHING! No need to care about the model.
    """
    load_path = os.path.join(opt.load_dir, opt.load_filename)
    """ 
    # old codes
    if opt.gpu_num > 1:#must add module at each parameter
        state_dict = torch.load(load_path, map_location=opt.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        state_dict = torch.load(load_path, map_location=opt.device)
    
    model.load_state_dict(state_dict)
    """
    model.load_state_dict(torch.load(load_path, map_location=opt.device))

    if opt.gpu_num > 1:
        model = nn.DataParallel(model)
    model = model.to(opt.device)
    
    return model