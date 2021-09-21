import h5py
import numpy as np
import torch
from torch._C import device
from OaR_segmentation.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter
from OaR_segmentation.preprocessing.preprocess_dataset import *
from torch.utils.data import Dataset


class HDF5lastlayer(Dataset):
    def __init__(self, scale: float, db_info: dict, mode: str, hdf5_db_dir: str, labels: dict,  
                 channels, augmentation=False, multiclass_test = False, db_set_train=False, lastlayer_fusion=False):
        self.db_info = db_info
        self.labels = labels
        self.db_dir = hdf5_db_dir
        self.scale = scale
        self.mode = mode
        self.augmentation = augmentation
        self.channels = channels
        self.multiclass_test = multiclass_test
        self.lastlayer_fusion = lastlayer_fusion

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        # used to train the convolutional stacking on train set, but with test set transformations
        if db_set_train:
            mode = 'train'
        
        self.ids_img = []
        self.ids_mask = []
        
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for volumes in db[f'{self.db_info["name"]}/{mode}'].keys():
            if volumes in [f'volume_{x}' for x in self.db_info['train_reduced']]:
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
        
        img_dict = {}
        mask_gt = np.zeros(shape=mask.shape, dtype=int)
        

        # Adjust the level of ct for coarse segmentation
        img_coarse = setDicomWinWidthWinCenter(img_data=img, winwidth=self.db_info["CTwindow_width"]["coarse"],
                                                wincenter=self.db_info["CTwindow_level"]["coarse"])
        img_coarse = np.uint8(img_coarse)
        
        if self.multiclass_test:
            mask_gt = mask
            #empty img_dict
        else:
            for key in self.labels.keys():
                img_single_organ = setDicomWinWidthWinCenter(img_data=img,
                                                            winwidth=self.db_info["CTwindow_width"][self.labels[key]],
                                                            wincenter=self.db_info["CTwindow_level"][self.labels[key]])
                img_single_organ = np.uint8(img_single_organ)
                img_single_organ = prepare_inference(img=img_single_organ, scale=self.scale)
                img_single_organ = torch.from_numpy(img_single_organ).type(torch.FloatTensor)
                img_dict[key] = img_single_organ
                
                # Create the ground truth mask 
                mask_gt[mask == int(key)] = key

        # Some preprocessing to the images
        img_coarse, mask_gt = prepare_inference(img=img_coarse, mask=mask_gt, scale=self.scale)

        return {
            'image_coarse': torch.from_numpy(img_coarse).type(torch.FloatTensor),
            'mask_gt': torch.from_numpy(mask_gt).type(torch.FloatTensor),
            'id': self.ids_img[idx],
            'dict_organs': img_dict
        }
