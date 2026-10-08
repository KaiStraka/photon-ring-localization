# KAI STRAKA 2026

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, gaussian_filter


def pixel_list(image):          # generate pixel list | use to load image

    data = np.load(image)
    pixel_values = data["I"].astype(np.float64)         # extract pixel values
    pixel_values /= pixel_values.max()      # normalize list

    return pixel_values


def pixel_tensor(image):        # generate pixel tensor

    pixels = torch.tensor(image, dtype = torch.float32)      # convert to tensor
    pixels = pixels.unsqueeze(0)        # match torch dimension

    return pixels


def ring_mask(pixels):      # segmented percentile

    percentile = 99.4                   # nth percentile pixels
    mask = np.zeros_like(pixels, dtype = bool)

    height, width = pixels.shape

    mid_y = height // 2
    mid_x = width // 2

    thresholds = np.array([                                         # quadrant thresholds for local percentiles
        [np.percentile(pixels[:mid_y, :mid_x], percentile),         # top left quadrant
         np.percentile(pixels[:mid_y, mid_x:], percentile)],        # top right quadrant
        [np.percentile(pixels[mid_y:, :mid_x], percentile),         # bottom left quadrant
         np.percentile(pixels[mid_y:, mid_x:], percentile)]])       # bottom right quadrant

    threshold_map = np.empty((height, width), dtype = float)

    threshold_map[:mid_y, :mid_x] = thresholds[0, 0]        # fill top left quad
    threshold_map[:mid_y, mid_x:] = thresholds[0, 1]        # fill top right quad
    threshold_map[mid_y:, :mid_x] = thresholds[1, 0]        # fill bottom left quad
    threshold_map[mid_y:, mid_x:] = thresholds[1, 1]        # fill bottom right quad

    threshold_map = gaussian_filter(threshold_map, sigma = 10.0, mode = "nearest")      # apply gaussian to thresholds
    mask = pixels >= threshold_map                                                      # apply threshold map to mask

    mask = binary_closing(mask, structure = np.ones((5, 5)))        # fill small gaps at borders
    mask = np.where(mask, pixels, 0.0)                              # preserve differences in brightness

    return mask


def display_img(cmap):      # plot images | internal testing only

    image_path = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\train\0jcnhtdsm2.npz"

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(pixel_list(image_path), origin = "lower", cmap = cmap)
    plt.colorbar(label = "Intensity")
    plt.title("Clean Image")

    plt.subplot(1, 2, 2)
    plt.imshow(ring_mask(pixel_list(image_path)), origin = "lower", cmap = cmap)
    plt.colorbar(label = "")
    plt.title("Clean Mask")

    plt.tight_layout()
    plt.show()


# display_img("afmhot")   # also "gray"