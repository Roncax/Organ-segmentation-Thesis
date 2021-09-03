import numpy as np
from numpy.core.numeric import full
from OaR_segmentation.network_architecture.net_factory import build_net
import json
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from OaR_segmentation.utilities.data_vis import visualize
from OaR_segmentation.inference.predictors.Predictor import Predictor
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset


class StandardPredictor(Predictor):
    def __init__(self, scale, mask_threshold,  paths, labels, n_classes):
        super(StandardPredictor, self).__init__(
            scale=scale, mask_threshold=mask_threshold,  paths=paths, labels=labels, n_classes=n_classes)
        self.net = None
        self.channels = None

    def initialize(self, channels, load_models_dir, models_type_list, deeplab_backbone):
        super(StandardPredictor, self).initialize()
        self.channels = channels
        self.net = self.initialize_net(
            load_models_dir=load_models_dir, models_type_list=models_type_list, deeplab_backbone=deeplab_backbone)

    def initialize_net(self, load_models_dir, models_type_list, deeplab_backbone):
        self.paths.set_pretrained_model(load_models_dir["coarse"])
        coarse_net = build_net(model=models_type_list["coarse"], n_classes=self.n_classes,
                               channels=self.channels, load_inference=True,
                               load_dir=self.paths.dir_pretrained_model, backbone=deeplab_backbone)
        return coarse_net

    def predict(self):
        super(StandardPredictor, self).predict()

        self.net.eval()
        dataset = HDF5Dataset(scale=self.scale, mode='test', db_info=json.load(open(self.paths.json_file_database)), hdf5_db_dir=self.paths.hdf5_db,
                              labels=self.labels, channels=self.net.n_channels, multiclass_test=True)
        test_loader = DataLoader(
            dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        with h5py.File(self.paths.hdf5_results, 'w') as db:
            with tqdm(total=len(dataset), unit='img') as pbar:
                for batch in test_loader:
                    imgs = batch['image_coarse'].to(device="cuda", dtype=torch.float32)
                    id = batch['id']

                    with torch.no_grad():
                        output = self.net(imgs)

                    if self.net.n_classes > 1:
                        # prob from 0 to 1 (dim = masks)
                        probs = F.softmax(output, dim=1)
                    else:
                        probs = torch.sigmoid(output)

                    full_mask = probs.squeeze().cpu().numpy()

                    if self.net.n_classes > 1:
                        res = self.combine_predictions(output_masks=np.delete(full_mask, 0, 0))
                    else:
                        full_mask = full_mask.squeeze()
                        res = full_mask > self.mask_threshold

                    # TESTING
                    # real_img = batch['image']
                    # real_img = real_img.squeeze().cpu().numpy()
                    # mask = batch['mask']
                    # mask = mask.squeeze().cpu().numpy()
                    # visualize(image=real_img, mask=mask, additional_1=res, additional_2=res)

                    db.create_dataset(id[0], data=res)
                    # update the pbar by number of imgs in batch
                    pbar.update(imgs.shape[0])
