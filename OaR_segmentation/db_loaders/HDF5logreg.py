import h5py
import numpy as np
from OaR_segmentation.preprocessing.preprocess_dataset import *
from torch.utils.data import Dataset
import torch


class HDF5DatasetLogReg(Dataset):
    def __init__(self, scale: float, db_info: dict, mode: str, hdf5_db_dir: str, labels: dict):
        self.db_info = db_info
        self.labels = labels
        self.db_dir = hdf5_db_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_mask = []
        
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for volumes in db[f'{self.db_info["name"]}/{mode}'].keys():
            ks = db[f'{self.db_info["name"]}/{mode}/{volumes}/image'].keys()
            for slice in ks:
                self.ids_mask.append(f'{self.db_info["name"]}/{mode}/{volumes}/mask/{slice}')


    def __len__(self):
        return len(self.ids_mask)

    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')
        mask = db[self.ids_mask[idx]][()]

        all_organ_masks = {}
        
        for key in self.labels.keys():
            mask_gt = np.zeros(shape=mask.shape, dtype=int)            
            mask_gt[mask == int(key)] = key
            mask_gt = prepare_inference(mask=mask_gt, scale=self.scale, normalize=False)
            mask_gt = torch.from_numpy(mask_gt).type(torch.FloatTensor)
            all_organ_masks[key] = mask_gt

        # Some preprocessing to the images
      
        return {
            'masks':all_organ_masks,
            'id': self.ids_mask[idx],
        }
