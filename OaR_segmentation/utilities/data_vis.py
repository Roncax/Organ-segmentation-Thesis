import json
import logging
import os

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from OaR_segmentation.evaluation import metrics


def save_img_mask_plot(img, mask, ground_truth, paths, fig_name="fig", patient_name="Default"):
    save_path = f"{paths.dir_plot_saves}/{patient_name}/{fig_name}"
    fig, ax = plt.subplots(1, 3)

    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].set_title('Output mask')
    ax[1].imshow(mask)
    ax[2].set_title('Ground truth')
    ax[2].imshow(ground_truth)
    ax[1].axis('off')
    ax[2].axis('off')

    os.makedirs(f"{paths.dir_plot_saves}/{patient_name}", exist_ok=True)
    plt.savefig(save_path + ".png")
    plt.close()


def prediction_plot(img, mask, ground_truth):
    fig, ax = plt.subplots(1, 3)

    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].axis('off')

    ax[2].set_title('Output mask')
    ax[2].imshow(mask)
    ax[2].axis('off')

    ax[1].set_title('Ground truth')
    ax[1].imshow(ground_truth)
    ax[1].axis('off')

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image


# create a gif from images of the target folder
def img2gif(png_dir, target_folder, out_name):
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(target_folder + f"/{out_name}.gif", images)

def volume2gif(volume, target_folder, out_name,):
    imageio.mimsave(f"{target_folder}/{out_name}.gif", volume)


def plot_single_result(score, type, paths, exp_num):
    fig, ax = plt.subplots()

    ax.boxplot(x=score.values(), labels=score.keys())
    plt.title(f"{exp_num}")
    plt.xticks(rotation=-45)
    plt.ylim(bottom=0)

    os.makedirs(f"{paths}", exist_ok=True)
    plt.savefig(paths + f"/{exp_num}_{type}.png")


def plot_results(results, paths, labels, used_net, met='all', mode=""):
    if met == all:
        met = metrics.ALL_METRICS.keys()

    for m in met:
        plot_single_result(results=results, type=m, paths=paths, labels=labels, mode=mode, used_net=used_net)


def visualize(image, mask, file_name=None, additional_1=None, additional_2=None ):
    fontsize = 18

    if additional_1 is None and additional_2 is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(image)
        ax[0, 0].set_title('Image', fontsize=fontsize)

        ax[1, 0].imshow(mask)
        ax[1, 0].set_title('Mask', fontsize=fontsize)

        ax[0, 1].imshow(additional_1)
        ax[0, 1].set_title('Additional 1', fontsize=fontsize)

        ax[1, 1].imshow(additional_2)
        ax[1, 1].set_title('Additional 2', fontsize=fontsize)

    plt.show()
    
    if file_name is not None:
        plt.savefig(f"{file_name}")
        plt.close()

def visualize_test(dict_images:dict, info:list=False):
    fontsize = 18

    f, ax = plt.subplots(len(dict_images), figsize=(8, 8))

    i=0
    for img in dict_images.keys():
        ax[i].imshow(dict_images[img])
        ax[i].set_title(img, fontsize=fontsize)
        i+=1
    # temp=""
    # for str_t in info:
    #     temp+= " " + str(str_t)
    # plt.figtext(0.5, 0.01, temp, ha="center", fontsize=18, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    plt.show()
    #plt.savefig(str(random.randint(1, 1000)))



