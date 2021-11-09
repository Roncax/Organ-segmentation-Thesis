import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from OaR_segmentation.preprocessing.preprocess_dataset import *


class HDF5DatasetStacking(Dataset):
    def __init__(self, scale: float, hdf5_db_dir, labels: dict,  channels, augmentation=False, crop_size=None):

        self.labels = labels
        self.db_dir = hdf5_db_dir
        self.scale = scale
        self.augmentation = augmentation 
        self.channels = channels
        self.crop_size = crop_size
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_mask = []
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for n in db.keys():
            self.ids_mask.append(n)


    def __len__(self):
        return len(self.ids_mask)


    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')
        masks = db[self.ids_mask[idx]] # "n/organ"
        
        final_array = None
        
        for label in self.labels.keys():
            gt_mask = prepare_inference(img=masks["gt"], scale=self.scale, normalize=False, crop_size=self.crop_size)

            if "coarse" in label:
                t_mask_dict = prepare_inference_multiclass(img=masks[label], scale=self.scale, normalize=False, crop_size=self.crop_size)
            else:
                t_mask_dict = prepare_inference(img=masks[label], scale=self.scale, normalize=False, crop_size=self.crop_size)
            
            
            if final_array is None:
                final_array = t_mask_dict
            else:
                final_array = np.concatenate((final_array, t_mask_dict), axis=0)
            
                
        return {
                    'image_coarse': torch.from_numpy(final_array).type(torch.FloatTensor),
                    'mask_gt': torch.from_numpy(gt_mask).type(torch.FloatTensor),
                    'id': self.ids_mask[idx]
                }