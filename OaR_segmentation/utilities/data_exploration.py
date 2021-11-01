import os
import sys

sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')
from OaR_segmentation.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter
from OaR_segmentation.preprocessing.preprocess_dataset import preprocess_test

from tqdm import trange
from os import path


from OaR_segmentation.db_loaders.HDF5Exploration import HDF5Exploration
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset

from OaR_segmentation.utilities.paths import Paths
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns


def find_HU_composition(dataset):
    gen = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    total_img = None
    with trange(len(gen), unit='batch', leave=False) as tbar:
        for batch in gen:            
            if total_img is None:
                total_img = batch['img']
                shape = np.shape(batch['img'])
            else:
                total_img+= batch['img']
            tbar.update(n=1)

    return total_img/(len(gen))


def plot_HU(tr_HU, test_HU, paths):
    data_tr=tr_HU.detach().cpu().numpy().flatten()
    data_test=test_HU.detach().cpu().numpy().flatten()
    sns.histplot(data=data_tr, color='blue', alpha=0.4, bins=70,binwidth = 1, kde=True)
    hist = sns.histplot(data=data_test, color='red', alpha=0.4, bins=70,binwidth = 1, kde=True)
    hist.set_xlim(left=-1050)
    hist.set_ylim(top = 4000)
    plt.xlabel('HU mean value')
    plt.ylabel('Pixel count')
    plt.savefig(paths.dir_database+"/HU_analysis.png")
    
def find_organ_percentage(dataset, labels):
    gen = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    
    organ_percent = None
    with trange(len(gen), unit='batch', leave=False) as tbar:
        for batch in gen:
            mask = batch['mask'].detach().squeeze().cpu().numpy()
            
            if organ_percent is None:
                organ_percent={}
                for i in labels.keys():
                    organ_percent[i] = np.count_nonzero(mask == int(i))
                organ_percent['0'] = np.count_nonzero(mask == 0)
            else:
                for i in labels.keys():
                    organ_percent[i] += np.count_nonzero(mask == int(i))
                organ_percent['0'] += np.count_nonzero(mask == 0)
                
            tbar.update(n=1)
        
        tot = 0
        for key, value in organ_percent.items():
            tot += value
            
        for key, value in organ_percent.items():
            organ_percent[key] = (value/tot)*100
            
    return organ_percent

def plot_organ_percentage(db, paths):
    
    plt.pie([db['0'],100-db['0'] ], labels=['Background', 'Others'], autopct='%1.1f%%')    
    plt.savefig(paths.dir_database+"/bg_percentage.png")
    plt.close()

    db.pop('0')    
    plt.pie(db.values(), labels=db.keys() , autopct='%1.1f%%')
    plt.pie(db.values(), labels=db.keys(), autopct='%1.1f%%')
    plt.savefig(paths.dir_database+"/Organ_percentage.png")
    plt.close()
    
    
def find_organ_number(dataset, labels):
    gen = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    
    organ_num = None
    with trange(len(gen), unit='batch', leave=False) as tbar:
        for batch in gen:
            mask = batch['mask'].squeeze()
            
            if organ_num is None:
                organ_num={}
                for i in labels.keys():
                    organ_num[i] = 1 if np.count_nonzero(mask == int(i)) > 0 else 0
            else:
                for i in labels.keys():
                    organ_num[i] += 1 if np.count_nonzero(mask == int(i)) > 0 else 0
            
            
            tbar.update(n=1)
    organ_num['total'] = len(gen)
    return organ_num

def plot_organ_number(tr, test, paths):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,10))
    fig.suptitle('Training and Test HU differences')
    
    ax1.bar(tr.keys() ,tr.values(),  color='c')

    ax1.set_title('Training dataset')
    
    ax2.bar(test.keys(), test.values(), color='c')
    plt.xlabel("Organ")
    plt.ylabel("Frequency")
    ax2.set_title('Test dataset')
    
    
    sns.barplot()

    #plt.show()
    plt.savefig(paths.dir_database+"/OrganxImage_analysis.png")
    plt.close()
    
def plot_img(db, paths):
    n_train = len(db)
    db_gen = DataLoader(db, batch_size=1, shuffle=True, num_workers=8, pin_memory=True,drop_last=True)
    
    for data in db_gen:
        slice = data['img'].squeeze()
        mask = data['mask'].squeeze()
        id_=data['id']
        
        s = id_[0].split("/")[-1]
        vol = id_[0].split("/")[2]
        
        if vol == 'volume_Patient_05':
            folder = f"{paths.dir_database}/cropped_images"
            
            slice = setDicomWinWidthWinCenter(img_data=slice, winwidth=1800, wincenter=-500)         
            slice = preprocess_test(img=slice, augmentation=False, crop_size=(320,320)).squeeze()

            os.makedirs(folder, exist_ok=True)
            plt.imshow(slice, cmap='gray')
            # plt.imshow(mask, cmap='nipy_spectral', alpha=0.5)
            # plt.colorbar()
            plt.axis('off')  # clear x-axis and y-axis
            plt.savefig(f"{folder}/{s}.png")
            plt.close()
            
            
            

if __name__ == '__main__':
    
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    platform = "local" #local, gradient, polimi
    paths = Paths(db=db_name, platform=platform)
    labels = {
"1": "RightLung",
"2": "LeftLung",
"3": "Heart",
"4": "Trachea",
"5": "Esophagus",
"6": "SpinalCord"

    }
    
    tr_dataset = HDF5Exploration(hdf5_db_dir=paths.hdf5_db, mode='train',  db_info=json.load(open(paths.json_file_database)))
    test_dataset = HDF5Exploration(hdf5_db_dir=paths.hdf5_db, mode='test',  db_info=json.load(open(paths.json_file_database)))
    all_db =  HDF5Exploration(hdf5_db_dir=paths.hdf5_db, mode='all',  db_info=json.load(open(paths.json_file_database)))

    # HU analysis
    # training_HU = find_HU_composition(tr_dataset)
    # test_HU = find_HU_composition(test_dataset)
    # plot_HU(training_HU, test_HU, paths=paths)
    
    # % organ analysis
    perc = find_organ_percentage(tr_dataset, labels=labels)
    plot_organ_percentage(perc, paths=paths)
    
    #plot_img(all_db, paths)
    