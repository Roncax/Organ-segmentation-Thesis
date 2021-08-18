import json
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from OaR_segmentation.inference.predictors.Predictor import Predictor
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset

class StandardPredictor(Predictor):
    def __init__(self):
        super(StandardPredictor, self).__init__()
        self.net = None
        self.channels = None
        
        
    def initialize(self, channels, load_models_dir, models_type_list, deeplab_backbone):
        super(StandardPredictor, self).initialize(channels, load_models_dir, models_type_list, deeplab_backbone)
        self.channels = channels
        self.net = self.initialize_net(load_models_dir=load_models_dir, models_type_list=models_type_list
                                       , deeplab_backbone=deeplab_backbone)
        
        
    def initialize_net(self, load_models_dir, models_type_list, deeplab_backbone):
        self.paths.set_pretrained_model(load_models_dir["coarse"])
        coarse_net = self.build_net(model=models_type_list["coarse"], n_classes=self.n_classes, 
                                    channels=self.channels, load_inference=True,
                                    load_dir=self.paths.dir_pretrained_model, backbone=deeplab_backbone)
        return coarse_net
    
        
    def predict(self):
        super(StandardPredictor, self).predict()
        
        self.net.eval()
        dataset = HDF5Dataset(scale=self.scale, mode='test', db_info=json.load(open(self.paths.json_file_database)), hdf5_db_dir=self.paths.hdf5_db,
                            labels=self.labels, channels=self.net.n_channels)
        test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        with h5py.File(self.paths.hdf5_results, 'w') as db:
            with tqdm(total=len(dataset), unit='img') as pbar:
                for batch in test_loader:
                    imgs = batch['image'].to(device="cuda", dtype=torch.float32)
                    id = batch['id']

                    with torch.no_grad():
                        output = self.net(imgs)

                    if self.net.n_classes > 1:
                        probs = F.softmax(output, dim=1)  # prob from 0 to 1 (dim = masks)
                    else:
                        probs = torch.sigmoid(output)

                    full_mask = probs.squeeze().cpu().numpy()                    
                    
                    if self.net.n_classes > 1:
                        res = self.combine_predictions(output_masks=full_mask)
                    else:
                        full_mask = full_mask.squeeze()
                        res = full_mask > self.mask_threshold

                    db.create_dataset(id[0], data=res)
                    pbar.update(imgs.shape[0])  # update the pbar by number of imgs in batch