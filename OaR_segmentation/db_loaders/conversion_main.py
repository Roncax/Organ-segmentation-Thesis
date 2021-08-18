import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/') # gradient: Organ-segmentation-Thesis

import OaR_segmentation.utilities.paths as paths
from dicom_load import prepare_dicom

if __name__ == '__main__':
    prepare_dicom(paths=paths.Paths(db="SegTHOR", platform="local"), split_train_test=False)
