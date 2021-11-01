import h5py
import numpy as np
import torch
from torch._C import device
from OaR_segmentation.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter
from OaR_segmentation.preprocessing.preprocess_dataset import *
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, scale: float, db_info: dict, mode: str, hdf5_db_dir: str, labels: dict,  
                 channels, augmentation=False, multiclass_test = False, db_set_train=False, train_with_reduced_db=False, crop_size=None):
        self.db_info = db_info
        self.labels = labels
        self.db_dir = hdf5_db_dir
        self.scale = scale
        self.mode = mode
        self.augmentation = augmentation
        self.channels = channels
        self.multiclass_test = multiclass_test
        self.train_with_reduced_db = train_with_reduced_db
        self.crop_size = crop_size

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        # used to train the convolutional stacking on train set, but with test set transformations
        if db_set_train:
            mode = 'train'
        
        self.ids_img = []
        self.ids_mask = []
        self.ids_train_reduced = []
        self.ids_mask_train_reduced = []
        
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for volumes in db[f'{self.db_info["name"]}/{mode}'].keys():
            
            ks = db[f'{self.db_info["name"]}/{mode}/{volumes}/image'].keys()
            for slice in ks:
                self.ids_img.append(f'{self.db_info["name"]}/{mode}/{volumes}/image/{slice}')
                self.ids_mask.append(f'{self.db_info["name"]}/{mode}/{volumes}/mask/{slice}')
                if volumes in [f'volume_{x}' for x in self.db_info['train_reduced']]:
                    self.ids_train_reduced.append(f'{self.db_info["name"]}/{mode}/{volumes}/image/{slice}')
                    self.ids_mask_train_reduced.append(f'{self.db_info["name"]}/{mode}/{volumes}/mask/{slice}')
                    
        assert len(self.ids_img) == len(self.ids_mask), f"Error in the number of mask {len(self.ids_mask)} and images{len(self.ids_img)}"

    def __len__(self):
        if self.train_with_reduced_db:
            length = len(self.ids_train_reduced)
        else:
            length = len(self.ids_img)
        
        return length

    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')

        
        if self.train_with_reduced_db:
            img = db[self.ids_train_reduced[idx]][()]
            mask = db[self.ids_mask_train_reduced[idx]][()]
        else:
            img = db[self.ids_img[idx]][()]
            mask = db[self.ids_mask[idx]][()]

        assert img.size == mask.size,f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        
        img_dict = {}
        mask_gt = np.zeros(shape=mask.shape, dtype=int)
        
        # TRAINING PREPROCESSING
        if self.mode == "train":
            if len(self.labels.items()) == 1:

                single_label = next(iter(self.labels))
                img_coarse = setDicomWinWidthWinCenter(img_data=img,
                                                winwidth=self.db_info["CTwindow_width"][self.labels[single_label]],
                                                wincenter=self.db_info["CTwindow_level"][self.labels[single_label]])
                mask_gt[mask == int(single_label)] = 1
                mask = mask_gt

            else:

                img_coarse = setDicomWinWidthWinCenter(img_data=img,
                                                winwidth=self.db_info["CTwindow_width"]["coarse"],
                                                wincenter=self.db_info["CTwindow_level"]["coarse"])

            img_coarse = np.uint8(img_coarse)
            img_coarse, mask_gt = preprocess_segmentation(img=img_coarse, mask=mask, scale=self.scale,
                                                          augmentation=self.augmentation, crop_size=self.crop_size)
        
        # TESTING and STACKING PREPROCESSING
        # Adjust the level of ct for fine segmentation (all organs in a labels)
        elif self.mode == "test":
            # Adjust the level of ct for coarse segmentation
            img_coarse = setDicomWinWidthWinCenter(img_data=img, winwidth=self.db_info["CTwindow_width"]["coarse"],
                                                   wincenter=self.db_info["CTwindow_level"]["coarse"])
            img_coarse = np.uint8(img_coarse)
            
            if self.multiclass_test:
                mask_gt = mask
                #empty img_dict
            else:
                for key in self.labels.keys():
                    
                    if '_' in key:
                        index = key.find('_')
                        k = key[:index]
                    else:
                        k = key
                        
                    img_single_organ = setDicomWinWidthWinCenter(img_data=img,
                                                                winwidth=self.db_info["CTwindow_width"][self.labels[k]],
                                                                wincenter=self.db_info["CTwindow_level"][self.labels[k]])
                    img_single_organ = np.uint8(img_single_organ)
                    img_single_organ = prepare_inference(img=img_single_organ, scale=self.scale, normalize=True, crop_size=self.crop_size)
                    img_single_organ = torch.from_numpy(img_single_organ).type(torch.FloatTensor)
                    img_dict[key] = img_single_organ
                    
                    # Create the ground truth mask 
                    #mask_gt[mask == int(key)] = key

            # Some preprocessing to the images
            mask_gt = mask
            img_coarse = prepare_inference(img=img_coarse, scale=self.scale, normalize=True, crop_size=self.crop_size)
            mask_gt = prepare_inference(img=mask_gt, scale=self.scale, normalize=False, crop_size=self.crop_size)

        return {
            'image_coarse': torch.from_numpy(img_coarse).type(torch.FloatTensor),
            'mask_gt': torch.from_numpy(mask_gt).type(torch.FloatTensor),
            'id': self.ids_img[idx] if not self.train_with_reduced_db else self.ids_train_reduced[idx],
            'dict_organs': img_dict
        }
