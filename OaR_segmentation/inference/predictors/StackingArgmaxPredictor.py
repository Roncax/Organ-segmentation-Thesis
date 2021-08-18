import json
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.inference.predictors.Predictor import Predictor
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset


class StackingArgmaxPredictor(Predictor):
    def __init__(self):
        super(StackingArgmaxPredictor, self).__init__()
        self.nets = None
        self.channels = None
        
        
    def initialize(self, channels, load_models_dir, models_type_list):
        super(StackingArgmaxPredictor, self).initialize(channels, load_models_dir, models_type_list)
        self.channels = channels
        self.nets = self.initialize_multinets(load_models_dir=load_models_dir, models_type_list= models_type_list)
    

    def initialize_multinets(self, load_models_dir, models_type_list):
        nets = {}
        for label in self.labels.keys():
            self.paths.set_pretrained_model(load_models_dir[label])

            nets[label] = build_net(model=models_type_list[label], n_classes=1, 
                                    channels=self.channels, load_inference=True,
                                    load_dir=self.paths.dir_pretrained_model)
        
        return nets
    
        
    def predict(self):
        super(StackingArgmaxPredictor, self).predict()
        
        dataset = HDF5Dataset(scale=self.scale, mode='test', db_info=json.load(open(self.paths.json_file_database)), paths=self.paths,
                            labels=self.labels)
        test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

        with h5py.File(self.paths.hdf5_results, 'w') as db:
            with tqdm(total=len(dataset), unit='img') as pbar:
                for batch in test_loader:
                    imgs = batch['image_organ']
                    id = batch['id']
                    final_array_prediction = None

                    for organ in self.nets.keys():
                        self.nets[organ].eval()
                        img = imgs[organ].to(device="cuda", dtype=torch.float32)

                        with torch.no_grad():
                            output = self.nets[organ](img)
                        
                        if final_array_prediction is None:
                            final_array_prediction = output
                        else:
                            final_array_prediction = torch.cat((output, final_array_prediction), dim=1)

                    probs = final_array_prediction
                    probs = torch.sigmoid(probs)
                    full_mask = probs.squeeze().cpu().numpy()
                    comb_img = self.combine_predictions(output_masks=full_mask)

                    db.create_dataset(id[0], data=comb_img) # add the calcualted image in the hdf5 results file
                    pbar.update(img.shape[0])   # update the pbar by number of imgs in batch