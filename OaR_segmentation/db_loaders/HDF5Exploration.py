import h5py
import numpy as np
import torch
from OaR_segmentation.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter
from OaR_segmentation.preprocessing.preprocess_dataset import *
from torch.utils.data import Dataset


class HDF5Exploration(Dataset):
    def __init__(self,  db_info: dict, mode: str, hdf5_db_dir: str):
        self.db_info = db_info
        self.db_dir = hdf5_db_dir
        self.mode = mode


        self.ids_img = []
        self.ids_mask = []
        
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for volumes in db[f'{self.db_info["name"]}/{mode}'].keys():
            ks = db[f'{self.db_info["name"]}/{mode}/{volumes}/image'].keys()
            for slice in ks:
                self.ids_img.append(f'{self.db_info["name"]}/{mode}/{volumes}/image/{slice}')
                self.ids_mask.append(f'{self.db_info["name"]}/{mode}/{volumes}/mask/{slice}')

        assert len(self.ids_img) == len(self.ids_mask), f"Error in the number of mask {len(self.ids_mask)} and images{len(self.ids_img)}"

    def __len__(self):
        return len(self.ids_img)

    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')

        img = db[self.ids_img[idx]][()]
        mask = db[self.ids_mask[idx]][()]

        assert img.size == mask.size,f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        
        return {
            'img':img,
            'mask': mask,
            'id': self.ids_img[idx],
        }
