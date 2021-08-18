from OaR_segmentation.utilities.data_vis import visualize, visualize_test
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from OaR_segmentation.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter
from OaR_segmentation.preprocessing.prepare_augment_dataset import *


class HDF5Dataset(Dataset):
    def __init__(self, scale: float, db_info: dict, mode: str, hdf5_db_dir: str, labels: dict,  channels, augmentation=False):
        self.db_info = db_info
        self.labels = labels
        self.db_dir = hdf5_db_dir
        self.scale = scale
        self.mode = mode
        self.augmentation = augmentation
        self.channels = channels
        self.ids_img_mask_dict = []  # for multi-channel purpose
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_img = []
        self.ids_mask = []
        
        if mode == "stacking": mode="train" #only the local variable change, the class still see "stacking"
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for volumes in db[f'{self.db_info["name"]}/{mode}'].keys():
            ks = db[f'{self.db_info["name"]}/{mode}/{volumes}/image'].keys()
            for slice in ks:
                self.ids_img.append(f'{self.db_info["name"]}/{mode}/{volumes}/image/{slice}')
                self.ids_mask.append(f'{self.db_info["name"]}/{mode}/{volumes}/mask/{slice}')

                if self.channels > 1:
                    assert int(self.channels) % 2 == 1, f'The channels must be odd in number, but here are even'

                    t = (self.channels - 1) / 2
                    temp_list = []
                    for n in range(int(-t), int(t + 1)):
                        slice_number = int(slice.replace('slice_', ''))
                        s = str(int(slice_number) + n)
                        if s in ks:
                            temp_list.append(f'{self.db_info["name"]}/{mode}/{volumes}/image/slice_{s}')
                        else:
                            temp_list.append(f'{self.db_info["name"]}/{mode}/{volumes}/image/{slice}')

                        self.ids_img_mask_dict.append(temp_list)

        assert len(self.ids_img) == len(
            self.ids_mask), f"Error in the number of mask {len(self.ids_mask)} and images{len(self.ids_img)}"

        logging.info(f'Creating {self.mode} dataset with {len(self.ids_img)} images')

    def __len__(self):
        return len(self.ids_img)

    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')

        img = db[self.ids_img[idx]][()]
        mask = db[self.ids_mask[idx]][()]
        adjacents = []


        # if self.channels > 1:
        #     assert self.channels % 2 == 1, f'The channels must be odd in number, but here are even'
        #     for i in range(self.channels):
        #         adjacents.append(db[self.ids_img_mask_dict[idx][i]][()])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        if self.mode == "train":
            #if self.channels == 1:
                # only specified classes are considered in the DB
            l = [x for x in self.labels.keys() if x != str(0)]

            if len(l) == 1:
                mask_cp = np.zeros(shape=mask.shape, dtype=int)
                img = setDicomWinWidthWinCenter(img_data=img,
                                                winwidth=self.db_info["CTwindow_width"][self.labels[l[0]]],
                                                wincenter=self.db_info["CTwindow_level"][self.labels[l[0]]])
                mask_cp[mask == int(l[0])] = 1
                mask = mask_cp

                # print(f"MASK: {np.unique(mask)}")
                # print(f" {np.shape(mask)}")

            else:
                img = setDicomWinWidthWinCenter(img_data=img,
                                                winwidth=self.db_info["CTwindow_width"]["coarse"],
                                                wincenter=self.db_info["CTwindow_level"]["coarse"])

            img = np.uint8(img)
            img, mask = prepare_segmentation_img_mask(img=img, mask=mask, scale=self.scale,
                                                      augmentation=self.augmentation)
            #visualize(image=img.squeeze(), mask=mask.squeeze())



            # else:
            #     # only specified classes are considered in the DB
            #     l = [x for x in self.labels.keys() if x != str(0)]
            #     t1 = []
            #     if len(l) == 1:
            #         mask_cp = np.zeros(shape=mask.shape, dtype=int)
            #         for im in adjacents:
            #             img = setDicomWinWidthWinCenter(img_data=im,
            #                                             winwidth=self.db_info["CTwindow_width"][self.labels[l[0]]],
            #                                             wincenter=self.db_info["CTwindow_level"][self.labels[l[0]]])
            #             t1.append(img)
            #         mask_cp[mask == int(l[0])] = 1
            #         mask = mask_cp
            #
            #     else:
            #         for im in adjacents:
            #             img = setDicomWinWidthWinCenter(img_data=im,
            #                                             winwidth=self.db_info["CTwindow_width"]["coarse"],
            #                                             wincenter=self.db_info["CTwindow_level"]["coarse"])
            #             t1.append(img)
            #
            #     full_channels_img = np.empty(shape=(1, 512, 512)) #todo change parametric shape
            #
            #     for im in t1:
            #         img = np.uint8(im)
            #         img, mask = prepare_segmentation_img_mask(img=img, mask=mask, scale=self.scale,
            #                                                   augmentation=self.augmentation)
            #
            #         full_channels_img = numpy.append(full_channels_img, img, axis=0)
            #         print(np.shape(full_channels_img))
            #
            #
            #
            #     img = full_channels_img
                
            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'id': self.ids_img[idx],
            }

        # binary mask + multiclass mask and img
        elif self.mode == "test":
            img_dict = {}

            for lab in self.labels:
                img_temp = img.copy()
                img_temp = setDicomWinWidthWinCenter(img_data=img_temp,
                                                     winwidth=self.db_info["CTwindow_width"][self.labels[lab]],
                                                     wincenter=self.db_info["CTwindow_level"][self.labels[lab]])
                img_temp = np.uint8(img_temp)

                # img_temp = prepare_img(img_temp, self.scale)
                img_temp = prepare_segmentation_inference_single(img=img_temp, scale=self.scale)

                img_temp = torch.from_numpy(img_temp).type(torch.FloatTensor)
                img_dict[lab] = img_temp

            l = [x for x in self.labels.keys() if x != str(0)]
            mask_cp = np.zeros(shape=mask.shape, dtype=int)
            img = setDicomWinWidthWinCenter(img_data=img, winwidth=self.db_info["CTwindow_width"]["coarse"],
                                            wincenter=self.db_info["CTwindow_level"]["coarse"])
            for key in l:
                mask_cp[mask == int(key)] = key
            mask = mask_cp

            img = np.uint8(img)
            img, mask = prepare_segmentation_inference(img=img, mask=mask, scale=self.scale)

            
            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'id': self.ids_img[idx],
                'image_organ': img_dict
            }
            
        #! Attenzione scale non implementato ancora
        elif self.mode == "stacking":
            mask_dict = {}
            img_dict = {}


            l = [x for x in self.labels.keys() if x != str(0)]
 
            mask_gt = np.zeros(shape=mask.shape, dtype=int)
            for key in l:
                img_temp = img.copy()
                img_temp = setDicomWinWidthWinCenter(img_data=img_temp,
                                                     winwidth=self.db_info["CTwindow_width"][self.labels[key]],
                                                     wincenter=self.db_info["CTwindow_level"][self.labels[key]])
                img_temp = np.uint8(img_temp)

                # img_temp = prepare_img(img_temp, self.scale)
                img_temp = prepare_segmentation_inference_single(img=img_temp, scale=self.scale)

                img_temp = torch.from_numpy(img_temp).type(torch.FloatTensor)
                img_dict[key] = img_temp
                mask_gt[mask == int(key)] = key
                # temp_mask = np.zeros(shape=mask.shape, dtype=int)
                # temp_mask[mask == int(key)] = 1
                # temp_mask = prepare_segmentation_mask(mask=temp_mask, scale=self.scale)
                # temp_mask = np.uint8(temp_mask)
                # temp_mask = torch.from_numpy(temp_mask).type(torch.FloatTensor)
                # mask_dict[key] = temp_mask
                
            mask_gt = prepare_segmentation_mask(mask=mask_gt, scale=self.scale)
            mask_gt = np.uint8(mask_gt)

            return {
                'img_dict': img_dict,
                'mask_gt': torch.from_numpy(mask_gt).type(torch.FloatTensor),
                'id': self.ids_img[idx],
            }

