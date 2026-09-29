# KAI STRAKA 2026

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate

import photonImage


def gaus_blur(image):       # elliptical gaussian blur
    
    sd_maj = 10
    sd_min = sd_maj * 0.8
    angle_deg = 90
    img_rotate = rotate(image, angle_deg, reshape = False, order = 1)           # align axis with elliptical gaussian
    
    img_blur = gaussian_filter(img_rotate, (sd_maj, sd_min))            # apply gaussian
    img_blur = rotate(img_blur, - angle_deg, reshape = False, order = 1)        # rotate back

    return img_blur     # gaussian image, gauss standard deviation int


def scatter_screen(image):      # random interstellar scattering

    pixels = 256                                # image pixel size
    field = np.sqrt(image).astype(complex)      # approximate electric field for screen

    fx = np.fft.fftfreq(pixels)                 # converting into fourier space
    fy = np.fft.fftfreq(pixels)
    freq_fx, freq_fy = np.meshgrid(fx, fy)      # spatial frequencies

    aniso_factor = 3.5                          # creates anisotropic scattering -> vertical stretch
    freq_mag = np.sqrt((freq_fx ** 2) + ((aniso_factor * freq_fy) ** 2))     # magnitude of spatial frequencies x, y
    freq_mag[0, 0] = 1e-10                      # avoid division by 0 at origin
    power_spec = freq_mag ** (-11/6)            # kolmogorov phase screen

    noise = (np.random.normal(size = (pixels, pixels)) + \
             (1j * np.random.normal(size = (pixels, pixels))))      # random complex coefficients

    phase_ft = noise * power_spec               # phase fourier transform
    phase = np.fft.ifft2(phase_ft).real         # convert to real phase screen
    phase = gaussian_filter(phase, 1.5)         # gaussian smooths scattering by factor (phase, X)
    phase -= phase.mean()
    phase /= phase.std()
    phase *= 3.0                                # scattering strength

    field *= np.exp(1j * phase)         # apply phase screen | wave equation Ae^(im * phi)
    field_ft = np.fft.fft2(field)       # fourier transform

    prop_dist = 15                      # blurring parameter
    prop = np.exp(-1j * np.pi * prop_dist * ((freq_fx ** 2) + (freq_fy ** 2)))       # fresnel propogation factor
    field_ft *= prop

    field = np.fft.ifft2(field_ft)           # convert to image space
    scatter_image = np.abs(field) ** 2       # convert field to intensity values
    scatter_image /= scatter_image.max()     # normalize

    return scatter_image


def degraded_image(image_path):

    deg_image = scatter_screen(gaus_blur(photonImage.pixel_list(image_path)))

    return deg_image


def display_img(gaus):      # plot images | internal testing only

    path = r"C:\Users\Kai\Desktop\astroAI\blackholeML\model\train\0ayg0903fy.npz"        # individual image path | testing use only

    plt.figure(figsize = (12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(photonImage.pixel_list(path), origin = "lower", cmap = "afmhot")
    plt.colorbar(label = "Intensity")
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(gaus_blur(photonImage.pixel_list(path)), origin = "lower", cmap = "afmhot", vmax = 0.35)       # vmax to adjust brightness for plots 2, 3
    plt.colorbar(label = "Intensity")
    plt.title(f"Gauss{gaus}")

    plt.subplot(1, 3, 3)
    plt.imshow(degraded_image(path), origin = "lower", cmap = "afmhot", vmax = 1.4)
    plt.colorbar(label = "Intensity")
    plt.title(f"Gauss{gaus} + Scatter")

    plt.tight_layout()
    plt.show()


# display_img(10)