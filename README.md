# Organ at risk Segmentation - **WIP**

PyTorch implementation of multiple nets for image semantic segmentation with medical images (CT-MRI). The main work here is to find and compare optimal stacking methods for multiclass (and multi-dataset) segmentation.

## Segmentation Nets
- Unet
- SegNet
- DeepLabv3
- Se-ResUnet
## Stacking methods
- Convolutional
- Avg
## Datasets:
- Structseg 2019
- AAPM Lung 2017 (in progress)
- SegThor 2019

## Notes:
- nifti -> HDF5 translation

## DB directories:
```bash
data
  |- datasets
    |- database_name
      |- plots
      |- raw_data (.nii or dicom files)
        |- volume_number (patient)
      |- db_name.json (info about dataset)
  |- checkpoints_segmentation
  |- checkpoints_stacking
  |- results
  ```

## HDF5 structure:
```bash
dataset_name
  |- train
    |- volume_number (patient)
      |- image
        |-slice_number
      |- mask
        |-slice_number
  |- test
    |- volume_number (patient)
      |- image
        |-slice_number 
      |- mask
        |-slice_number
```
