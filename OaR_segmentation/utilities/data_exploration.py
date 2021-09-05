from os import path
import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')


from OaR_segmentation.db_loaders.HDF5Exploration import HDF5Exploration
from OaR_segmentation.utilities.paths import Paths
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

db_name = "StructSeg2019_Task3_Thoracic_OAR"
platform = "local" #local, gradient, polimi
paths = Paths(db=db_name, platform=platform)

# DATASET split train/val
dataset = HDF5Exploration(hdf5_db_dir=paths.hdf5_db, mode='train',  db_info=json.load(open(paths.json_file_database)))
tr_gen = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

for batch in tr_gen:
    plt.hist([batch['img'].flatten()], bins=100, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()
