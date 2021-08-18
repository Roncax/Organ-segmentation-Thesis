import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

from OaR_segmentation.utilities.data_vis import visualize, visualize_test
from OaR_segmentation.preprocessing.prepare_augment_dataset import *


class HDF5DatasetStacking(Dataset):
    def __init__(self, scale: float, hdf5_db_dir, labels: dict,  channels, augmentation=False,):

        self.labels = labels
        self.db_dir = hdf5_db_dir
        self.scale = scale
        self.augmentation = augmentation
        self.channels = channels
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_mask = []
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for n in db.keys():
            self.ids_mask.append(n)


        logging.info(f'Creating stacking dataset with {len(self.ids_mask)} images')

    def __len__(self):
        return len(self.ids_mask)

    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')
        masks = db[self.ids_mask[idx]]

        final_array = None
        for mask_name in masks.keys():
            
            if mask_name != "gt":
                
                t_mask_dict = prepare_segmentation_mask(mask=masks[mask_name], scale=self.scale)
                if final_array is None:
                    
                    
                    final_array = t_mask_dict
                else:
                    final_array = np.concatenate((t_mask_dict, final_array), axis=0)
            else:
                temp_mask = prepare_segmentation_mask(mask=masks[mask_name], scale=self.scale)
                gt_mask = temp_mask
            
                
        
        return {
                    'image': torch.from_numpy(final_array).type(torch.FloatTensor),
                    'mask': torch.from_numpy(gt_mask).type(torch.FloatTensor),
                    'id': self.ids_mask[idx]
                }