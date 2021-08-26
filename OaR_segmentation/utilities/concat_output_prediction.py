
import h5py
import torch
from torch.utils.data.dataloader import DataLoader
import tqdm
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset


def create_combined_dataset(scale, nets,  paths, labels):
    dataset = HDF5Dataset(scale=scale, mode='test', db_info=json.load(open(paths.json_file_database)), 
                          hdf5_db_dir=paths.hdf5_db, channels=1, labels=labels)
    
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    with h5py.File(paths.hdf5_stacking, 'w') as db:
        with tqdm(total=len(dataset), unit='img') as pbar:
            for i, batch in enumerate(test_loader):
                imgs = batch['dict_organs']
                mask_gt = batch['mask_gt']
                
                mask_gt = mask_gt.squeeze().cpu().numpy()
                db.create_dataset(f"{i}/gt", data=mask_gt)

                for organ in nets.keys():
                    nets[organ].eval()
                    img = imgs[organ].to(device="cuda", dtype=torch.float32)

                    with torch.no_grad():
                        output = nets[organ](img)

                    probs = output
                    probs = torch.sigmoid(probs) # todo log_softmax, raw logits, log_sigmoid, softmax
                    full_mask = probs.squeeze().squeeze().cpu().numpy()

                    # TESTING
                    # img_t = img.clone().detach().squeeze().cpu().numpy()
                    # full_mask_thresholded = full_mask > mask_threshold
                    # print(organ)
                    # visualize(image=full_mask, mask=img_t, additional_1=full_mask_thresholded, additional_2=mask_gt ,file_name=f"{i}_{organ}")
                    # if(i==0 and organ == "3"):
                    #     print("hey")
                    # res = np.array(full_mask).astype(np.bool)

                    db.create_dataset(f"{i}/{organ}", data=full_mask)

                # update the pbar by number of imgs in batch
                pbar.update(img.shape[0])