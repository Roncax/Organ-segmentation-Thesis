import json
import os
import random
import cv2
import numpy as np
from nibabel import load as load_nii
from preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter


# read and decode a .nii file
def read_nii(path, dataset_parameters):
    img = load_nii(path).get_fdata()
    img = setDicomWinWidthWinCenter(img, winwidth=dataset_parameters['CTwindow_width']['coarse'],
                                    wincenter=dataset_parameters['CTwindow_level']['coarse'])
    img = np.uint8(img)
    return img


# load all labels in path from a  .nii
def nii2label(nii_root, root_path, dataset_parameters, paths):
    names = [name for name in os.listdir(nii_root)]
    os.makedirs(root_path, exist_ok=True)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in dataset_parameters['test']:
            target_path = f'{paths.dir_test_GTimg}/{name}'
            os.makedirs(target_path, exist_ok=True)

        label_array = np.uint8(load_nii(f'{nii_path}/GT.nii').get_fdata())

        # save labels with patient's number
        for n in range(label_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        label_array[:, :, n])


# load all images in path from a  .nii
def nii2img(nii_root, root_path, dataset_parameters, paths):
    names = [name for name in os.listdir(nii_root)]

    os.makedirs(root_path, exist_ok=True)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in dataset_parameters['test']:
            target_path = f'{paths.dir_test_img}/{name}'
            os.makedirs(target_path, exist_ok=True)

        image_array = read_nii(os.path.join(nii_path, f"{name}.nii"), dataset_parameters)
        for n in range(image_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        image_array[:, :, n])


def random_split_test(dir, dataset_parameters, paths):
    names = [name for name in os.listdir(dir)]
    # choose random test images and populate the json file
    test_img = random.sample(names, dataset_parameters['numTest'])
    dataset_parameters['test'] = test_img
    dataset_parameters['train'] = [item for item in names if not item in test_img]
    assert len(dataset_parameters['train']) == dataset_parameters['numTraining'], f'Invalid number of train items'
    json.dump(dataset_parameters, open(paths.json_file, "w"))


def prepare_segthor(paths):
    dataset_parameters = json.load(open(paths.json_file))

    random_split_test(dir=paths.dir_raw_db, dataset_parameters=dataset_parameters, paths=paths)
    nii2img(nii_root=paths.dir_raw_db, root_path=paths.dir_train_imgs, dataset_parameters=dataset_parameters,
            paths=paths)
    nii2label(nii_root=paths.dir_raw_db, root_path=paths.dir_train_masks, dataset_parameters=dataset_parameters,
              paths=paths)
