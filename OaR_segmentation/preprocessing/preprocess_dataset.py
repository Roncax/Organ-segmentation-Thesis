from typing import final
import numpy as np
import albumentations as A
from numpy.lib.type_check import imag


def preprocess_segmentation(img, mask, scale, augmentation, crop_size = None):

    """Resize, augmentation, normalization of the image and transposition of dimension for numpy -> torch conversion

    Returns:
        img, mask: image and mask preprocessed
    """
    
    img = np.expand_dims(img, axis=2)
    mask = np.expand_dims(mask, axis=2)
    
    
    if crop_size is not None:
        crop = A.CenterCrop(height=crop_size[0], width=crop_size[1], always_apply=True)
        crop_img =  crop(image = img, mask = mask)
        img = crop_img['image']
        mask = crop_img['mask']
    
    if scale < 1:
        shape = np.shape(img)
        w = shape[0]
        h = shape[1]
        resize = A.Resize(height=int(scale * w), width=int(scale * h), always_apply=True)
        resized_img = resize(image=img, mask=mask)
        img = resized_img['image']
        mask = resized_img['mask']
    
    if augmentation:
        transform = A.Compose([
            A.ElasticTransform(p=0.5, alpha=120 * 0.25, sigma=120 * 0.05, alpha_affine=120 * 0.05),
            A.GridDistortion(p=0.5),
            #A.RandomScale(scale_limit=0.05, p=0.5),
            A.Rotate(limit=10, p=0.5),
            # A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=10, p=0.5),
            # A.Blur(blur_limit=7, always_apply=False, p=0.5),
            #A.GaussNoise(var_limit=(0, 10), p=0.5),
        ])

        transformed = transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        

    img = img / 255
    img = img.transpose((2, 0, 1))
    mask = mask.transpose((2, 0, 1))
    return img, mask

def preprocess_test(img, augmentation=False,  crop_size = None):

    """Resize, augmentation, normalization of the image and transposition of dimension for numpy -> torch conversion

    Returns:
        img, mask: image and mask preprocessed
    """
    
    img = np.expand_dims(img, axis=2)
    
    
    if crop_size is not None:
        crop = A.CenterCrop(height=crop_size[0], width=crop_size[1], always_apply=True)
        crop_img =  crop(image = img)
        img = crop_img['image']

    
    if augmentation:
        transform = A.Compose([
            A.ElasticTransform(p=0.5, alpha=120 * 0.25, sigma=120 * 0.05, alpha_affine=120 * 0.05),
            A.GridDistortion(p=0.5),
            #A.RandomScale(scale_limit=0.05, p=0.5),
            A.Rotate(limit=10, p=0.5),
            # A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=10, p=0.5),
            # A.Blur(blur_limit=7, always_apply=False, p=0.5),
            #A.GaussNoise(var_limit=(0, 10), p=0.5),
        ])

        transformed = transform(image=img)
        img = transformed['image']
        

    # img = img / 255
    # img = img.transpose((2, 0, 1))
    # mask = mask.transpose((2, 0, 1))
    return img


def prepare_inference(img, normalize, scale = 1, crop_size=None):
    """Same as above but no transformations
    """
    img = np.expand_dims(img, axis=2)
    
    if crop_size is not None:
        crop = A.CenterCrop(height=crop_size[0], width=crop_size[1], always_apply=True)
        crop_img =  crop(image = img)
        img = crop_img['image']
    
    if scale < 1:
        w, h = np.shape(img)
        resize = A.Resize(height=int(scale * w), width=int(scale * h), always_apply=True)
        resized_img = resize(image=img)
        img = resized_img['image']

    if normalize:
        img = img / 255
    img = img.transpose((2, 0, 1))


    return img


def prepare_inference_multiclass(img, normalize, scale = 1, crop_size=None):
    """Same as above but for multiclass
    """
    final_array=None
    if crop_size is not None:
        crop = A.CenterCrop(height=crop_size[0], width=crop_size[1], always_apply=True)

        for slice in img:
            crop_img =  crop(image = slice)
            t = crop_img['image']
            t = np.expand_dims(t, axis=0)
            
            if final_array is None:
                final_array = t
            else:
                final_array = np.concatenate((final_array, t), axis=0)
            
        img = final_array
    
    if scale < 1:
        w, h = np.shape(img)
        resize = A.Resize(height=int(scale * w), width=int(scale * h), always_apply=True)
        resized_img = resize(image=img)
        img = resized_img['image']

    if normalize:
        img = img / 255

    return img