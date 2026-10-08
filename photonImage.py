# KAI STRAKA 2026

import numpy as np
import torch
import matplotlib.pyplot as plt


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

    percentile = 99.5               # nth percentile pixels
    mask = np.zeros_like(pixels)

    height, width = pixels.shape

    mid_y = height // 2
    mid_x = width // 2

    block = pixels[:mid_y, :mid_x]      # top left quadrant
    mask[:mid_y, :mid_x] = block >= np.percentile(block, percentile)

    block = pixels[:mid_y, mid_x:]      # top right quadrant
    mask[:mid_y, mid_x:] = block >= np.percentile(block, percentile)

    block = pixels[mid_y:, :mid_x]      # bottom left quadrant
    mask[mid_y:, :mid_x] = block >= np.percentile(block, percentile)

    block = pixels[mid_y:, mid_x:]      # bottom right quadrant
    mask[mid_y:, mid_x:] = block >= np.percentile(block, percentile)

    return mask


def display_img():      # plot images | internal testing only

    image_path = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\train\00d2u2jywh.npz"

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(pixel_list(image_path), origin = "lower", cmap = "gray")
    plt.colorbar(label = "Intensity")
    plt.title("Clean Image")

    plt.subplot(1, 2, 2)
    plt.imshow(ring_mask(pixel_list(image_path)), origin = "lower", cmap = "gray")
    plt.colorbar(label = "")
    plt.title("Clean Mask")

    plt.tight_layout()
    plt.show()


display_img()