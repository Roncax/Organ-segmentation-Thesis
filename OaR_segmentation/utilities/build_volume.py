import json
import os
import numpy as np
from PIL import Image


def build_np_volume(dir):
    # TODO parametric shape
    volume = np.empty(shape=(512, 512, 1))
    in_files = os.listdir(dir)
    in_files.sort()  # if not sorted we cannot associate gt and prediction

    for i, fn in enumerate(in_files):
        img = Image.open(os.path.join(dir, fn))
        img = np.expand_dims(img, axis=2)
        volume = np.append(volume, img, axis=2).astype(dtype=int)

    return volume


def grayscale2rgb_mask(mask, colormap, labels):
    finalmask3D = np.empty(shape=(512, 512, 3))
    finalmask_r = np.empty(shape=(512, 512))
    finalmask_g = np.empty(shape=(512, 512))
    finalmask_b = np.empty(shape=(512, 512))

    for i in range(len(labels)):
        finalmask_r[mask == i] = colormap[str(i)][0]
        finalmask_g[mask == i] = colormap[str(i)][1]
        finalmask_b[mask == i] = colormap[str(i)][2]

    finalmask3D[:, :, 0] = finalmask_r
    finalmask3D[:, :, 1] = finalmask_g
    finalmask3D[:, :, 2] = finalmask_b

    return Image.fromarray(finalmask3D.astype(np.uint8))


def mask_to_image1D(mask):
    img = np.zeros((512, 512))
    for i, m in enumerate(mask):
        img[m] = i
    return Image.fromarray(img.astype(np.uint8))
