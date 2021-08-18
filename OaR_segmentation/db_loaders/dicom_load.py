import json
import os
import random
import cv2
import numpy as np
from nibabel import load as load_nii
import h5py


# load all labels in path from a  .nii
def nii2label(nii_root, root_path, dataset_parameters, paths):
    names = [name for name in os.listdir(nii_root)]
    os.makedirs(root_path, exist_ok=True)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in dataset_parameters['test']:
            target_path = f'{paths.dir_test_GTimg}/patient_{name}'
            os.makedirs(target_path)

        label_array = np.uint8(load_nii(f'{nii_path}/label.nii').get_fdata())

        # organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Trachea', 'Esophagus', 'SpinalCord']
        # label_array[label_array == 0] = 0
        # label_array[label_array == 1] = 1
        # label_array[label_array == 2] = 2
        # label_array[label_array == 3] = 3
        # label_array[label_array == 4] = 4
        # label_array[label_array == 5] = 5
        # label_array[label_array == 6] = 6

        # save labels with patient's number
        for n in range(label_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"patient_{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        label_array[:, :, n])


# load all images in path from a  .nii
def nii2img(nii_root, root_path, dataset_parameters, paths):
    names = [name for name in os.listdir(nii_root)]

    os.makedirs(root_path, exist_ok=True)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in dataset_parameters['test']:
            target_path = f'{paths.dir_test_img}/patient_{name}'
            os.makedirs(target_path, exist_ok=True)

        image_array = read_nii(os.path.join(nii_path, "data.nii"))
        for n in range(image_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"patient_{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        image_array[:, :, n])


# dims: (512,512,n_slice)
def nii_to_hdf5_img(nii_root: str, db_path: str, db_name: str, test_imgs: list):
    names = [name for name in os.listdir(nii_root)]

    with h5py.File(db_path, 'a') as f:
        for name in names:
            
            image_array = load_nii(os.path.join(nii_root, name, f"{name}.nii")).get_fdata()
            mask_array = load_nii(os.path.join(nii_root, name, f"GT.nii")).get_fdata()

            # save every img slice in a subdir
            for n in range(image_array.shape[2]):
                mode = "test" if name in test_imgs else "train"
                f.create_dataset(name=f'{db_name}/{mode}/volume_{name}/image/slice_{n}', data=image_array[:, :, n])
                f.create_dataset(name=f'{db_name}/{mode}/volume_{name}/mask/slice_{n}', data=mask_array[:, :, n])



    
# choose random n volumes and save names in json
def random_split_test(paths):
    db_info = json.load(open(paths.json_file_database))
    names = [name for name in os.listdir(paths.dir_raw_db)]
    test_img = random.sample(names, db_info['numTest'])
    db_info['test'] = test_img
    db_info['train'] = [item for item in names if not item in test_img]

    assert len(db_info['train']) == db_info['numTraining'], f'Invalid number of train items'

    json.dump(db_info, open(paths.json_file_database, "w"))


# final shape single dataset -> (512,512) mask and img
def prepare_dicom(paths, split_train_test=False):
    if split_train_test:
        random_split_test(paths=paths)
    db_name = json.load(open(paths.json_file_database))["name"]
    test_imgs = json.load(open(paths.json_file_database))["test"]
    nii_to_hdf5_img(db_path=paths.hdf5_db, db_name=db_name, nii_root=paths.dir_raw_db, test_imgs=test_imgs)
