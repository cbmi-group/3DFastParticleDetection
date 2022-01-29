import os
import mrcfile
import math
import random
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd

class Dataset_10045(Dataset):
    def __init__(self, mode='train', base_dir='/ldap_shared/synology_shared/10045/10045_icon_fit/bin4_iconmask2/', detect_size=40, padding=12, empty_num=0):
        """ Total cube size = detect_size + 2 * padding
            Make sure total cube size can be divided by 16 (for YOLO)
        """

        self.base_dir = base_dir
        self.mode = mode
        if self.mode == 'train':
            self.data_range = [6, 7, 8, 9]
        elif self.mode == 'val':
            self.data_range = [10]
        elif self.mode == 'test':
            self.data_range = [11]
        
        self.diam = 23
        self.detect_size = detect_size
        self.padding = padding
        self.cube_size = self.detect_size + 2 * self.padding
        self.empty_num = empty_num
        
        self.origin = [mrcfile.open(os.path.join(self.base_dir, 'data_std/IS002_291013_0%02d_iconmask2_norm_rot_cutZ.mrc' % i)).data for i in self.data_range]
        self.position = [pd.read_csv(os.path.join(self.base_dir, 'coords/IS002_291013_0%02d_iconmask2_norm_rot_cutZ.coords' % i), sep='\t', header=None).to_numpy() for i in self.data_range]

        #set up full volume and label for use
        self.full_volume = []
        for i in range(len(self.data_range)):
            d, h, w = self.origin[i].shape
            vol_w = math.ceil((w - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding
            vol_h = math.ceil((h - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding
            vol_d = math.ceil((d - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding

            volume = np.zeros((vol_d, vol_h, vol_w), dtype=np.float32)
            volume[:d, :h, :w] = self.origin[i]
            self.full_volume.append(volume)
        
        #set up label
        
        #set data and label for training and test
        self.data = []
        self.targets = []
        for i in range(len(self.data_range)):
            base_index = len(self.targets)

            #slices data
            data_vol = self.full_volume[i]
            len_x = (data_vol.shape[2] - 2 * self.padding) // self.detect_size
            len_y = (data_vol.shape[1] - 2 * self.padding) // self.detect_size
            len_z = (data_vol.shape[0] - 2 * self.padding) // self.detect_size

            for z in range(len_z):
                for y in range(len_y):
                    for x in range(len_x):
                        data_cube = data_vol[z * self.detect_size:(z + 1) * self.detect_size + 2 * padding, 
                                             y * self.detect_size:(y + 1) * self.detect_size + 2 * padding, 
                                             x * self.detect_size:(x + 1) * self.detect_size + 2 * padding]
                        self.data.append(data_cube)
                        self.targets.append(torch.rand(0, 6))
            
            #make label
            for particle in self.position[i]:
                x, y, z = particle[0:3]
                
                if x < self.padding or y < self.padding or z < self.padding:
                    continue

                t_index = base_index + (x - self.padding) // self.detect_size + (y - self.padding) // self.detect_size * len_x + (z - self.padding) // self.detect_size * len_x * len_y
                self.targets[t_index] = torch.cat((self.targets[t_index], torch.tensor([[(x - self.padding) % self.detect_size + self.padding, (y - self.padding) % self.detect_size + self.padding, (z - self.padding) % self.detect_size + self.padding, self.diam, 1.0, 1.0]])), dim=0)
        
        #clear data
        if self.mode == 'train':
            empty_list = []
            for i in range(len(self.data)):
                if len(self.targets[i]) == 0:
                    empty_list.append(i)
            
            #add empty data
            random.shuffle(empty_list)
            empty_list = empty_list[self.empty_num:]
            empty_list.sort(reverse=True)

            for i in empty_list:
                del self.data[i]
                del self.targets[i]
        
        print('total data: ', len(self.data))
        #print
        print('Setup ', mode, ' dataset ok.')
    

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        data = torch.tensor(np.array(data)).unsqueeze(0)

        return data, target
        
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        """
            Padding targets to same size for batching
        """
        imgs, targets = list(zip(*batch))
        imgs = torch.stack([img for img in imgs])

        max_num = max([boxes.size(0) for boxes in targets])
        if max_num == 0: #no object, make sure never occur
            padding_targets = torch.FloatTensor(imgs.size(0), 1, 6).fill_(0)
        else:
            padding_targets = torch.FloatTensor(imgs.size(0), max_num, 6).fill_(0)
            for i, boxes in enumerate(targets):
                padding_targets[i, :boxes.size(0), :] = boxes
        
        return imgs, padding_targets

    def joint(self, pred):
        """
            Must be in eval or test mode, and rotate is Fasle.
        """
        full_pred_list = []
        base_index = 0
        for i in range(len(self.data_range)):
            remove_count = 0
            volume_list = np.zeros((0, 6), dtype=np.int64)
            full_vol = self.full_volume[i]
            rec_vol = self.origin[i]

            len_x = (full_vol.shape[2] - 2 * self.padding) // self.detect_size
            len_y = (full_vol.shape[1] - 2 * self.padding) // self.detect_size
            len_z = (full_vol.shape[0] - 2 * self.padding) // self.detect_size

            for z in range(len_z):
                for y in range(len_y):
                    for x in range(len_x):
                        cube_index = base_index + x + y * len_x + z * len_x * len_y
                        #clear data
                        cube_list = pred[cube_index].numpy().round().astype(np.int64)
                        if len(cube_list) == 0:
                            continue
                        #remove padding
                        x_map = (cube_list[:, 0] >= self.padding) * (cube_list[:, 0] <= self.cube_size - self.padding)
                        y_map = (cube_list[:, 1] >= self.padding) * (cube_list[:, 1] <= self.cube_size - self.padding)
                        z_map = (cube_list[:, 2] >= self.padding) * (cube_list[:, 2] <= self.cube_size - self.padding)
                        cube_list = cube_list[x_map * y_map * z_map]
                        
                        cube_list += np.array([[x * self.detect_size, y * self.detect_size, z * self.detect_size, 0, 0, 0]])
                        #remove out point (important for evaluation)
                        x_map = cube_list[:, 0] < rec_vol.shape[2]
                        y_map = cube_list[:, 1] < rec_vol.shape[1]
                        z_map = cube_list[:, 2] < rec_vol.shape[0]
                        cube_list = cube_list[x_map * y_map * z_map]
                        if cube_list.shape[0] == 0:
                            continue
                        volume_list = np.concatenate((volume_list, cube_list), axis=0)
                        remove_count += len(pred[cube_index]) - len(cube_list)
            
            full_pred_list.append(volume_list)

            if remove_count > 0:
                print('Remove particle: ', remove_count)
            
            return full_pred_list