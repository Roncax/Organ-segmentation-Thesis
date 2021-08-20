import numpy as np
import albumentations as A


def preprocess_segmentation(img, mask, scale, augmentation):

    """Resize, augmentation, normalization of the image and transposition of dimension for numpy -> torch conversion

    Returns:
        img, mask: image and mask preprocessed
    """   
    w, h = np.shape(img)
    img = np.expand_dims(img, axis=2)
    mask = np.expand_dims(mask, axis=2)

    resize = A.Resize(height=int(scale * w), width=int(scale * h), always_apply=True)

    resized_img = resize(image=img, mask=mask)
    original_img = resized_img['image']
    original_mask = resized_img['mask']

    if augmentation:
        transform = A.Compose([
            A.ElasticTransform(p=0.5, alpha=120 * 0.25, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            #A.RandomScale(scale_limit=0.1, p=0.5),
            A.Rotate(limit=10, p=0.5),
            # A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=10, p=0.5),
            # A.Blur(blur_limit=7, always_apply=False, p=0.5),
            A.GaussNoise(var_limit=(0, 10), always_apply=False, p=0.5),
        ])

        transformed = transform(image=original_img, mask=original_mask)
        img = transformed['image']/255
        mask = transformed['mask']

    else:
        img = original_img / 255
        mask = original_mask

    img = img.transpose((2, 0, 1))
    mask = mask.transpose((2, 0, 1))
    return img, mask


def prepare_inference(img = None, mask=None, scale = 1):
    """Same as above but no transformations
    """        
    if mask is not None:
        w, h = np.shape(mask)
        mask = np.expand_dims(mask, axis=2)

        resize = A.Resize(height=int(scale * w), width=int(scale * h), always_apply=True)
        resized_img = resize(mask=mask)
        original_mask = resized_img['mask']
        mask = original_mask.transpose((2, 0, 1))    

    if img is not None:
        w, h = np.shape(img)

        img = np.expand_dims(img, axis=2)

        resize = A.Resize(height=int(scale * w), width=int(scale * h), always_apply=True)
        resized_img = resize(image=img)
        original_img = resized_img['image']

        img = original_img / 255
        img = img.transpose((2, 0, 1))

    return img, mask
