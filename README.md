KAI STRAKA 10.2026

Neural localization of n ≥ 0 photon ring structure in synthetically degraded black hole images.

## <brd>
**WORK IN PROGRESS PROJECT**

NOTE: this README is not a detailed description or representation of the project, consider it a temporary introduction.\
Below is brief information on current and ongoing results (all information IS subject to change). 

**INTRO**

As of October 2026, this purpose of this project is using a U-Net CNN (convolutional neural network) architecture for localization and ring mask prediction of n ≥ 0 photon ring structures within simulated and synthetically degraded black hole images.\
Below in Fig. 1 is an example of the current predicted ring mask pipeline in action.

<div align="center">
  <img width="1847" height="474" alt="github temp example" src="https://github.com/user-attachments/assets/74ae6a5c-8b7d-4cd6-b011-4924e2c2a0b3" />
  <p><em>Figure 1: Full prediction example (mask is outdated)</em></p>
</div>

## <brd>
**Image Degredation**

Simulated photon ring images all come from the GRRT<sup>[3]</sup><sup>[4]</sup> public dataset as clean and ideal images. My model currently uses two filters to degrade the images to realistic observational conditions.

The first layer is an elliptical Gaussian blur, currently set at a standard deviation of 10 (Gauss10)<sup>[4]</sup>. This was found to provide the most realistic blurring effect.\
The second layer is an interstellar scattering screen<sup>[1]</sup> using Kolmogorov turbulence.\
To achieve a degraded image, a clean image will first gain the Gaussian blur, then the scattering screen.\
Below in Fig. 2 is an example of the two degradation layers.

<div align="center">
  <img width="80%" height="573" alt="github temp example2" src="https://github.com/user-attachments/assets/d846c5ee-f06d-4fc6-a254-282b2aeee27d" />
  <p><em>Figure 2: (left to right) Clean simulated image, Gaussian blur, Scattering Screen and Gaussian</em></p>
</div>

## <brd>
**Ring Mask Generation**

A percentile method was initially used to find the 99.4%th percentile of brightest pixels from the clean image, and generate a binary mask from those pixels (note gradient masks are now implemented). This worked for ideal cases, but failed when the simulated black hole was asymmetric i.e one side of the ring is significantly brighter than the other. This fails because n = 0 layer pixels on the brighter half can be brighter than n = 1 layer pixels on the less bright half, leading to pixel flooding or just only partial mask generation, as can be seen in Fig. 3 below (left).

<div align="center">
  <img width="40%" height="475" alt="github temp example3" src="https://github.com/user-attachments/assets/0cbd8cb9-d8d0-4e74-810c-43583e7c4748" />
  <img width="41%" height="400" alt="github temp example5" src="https://github.com/user-attachments/assets/f62019e9-77d2-431d-9d0c-a4b3c7cffd97" />
  <p><em>Figure 3: Percentile only ring mask / Figure 4: Segmented ring mask</em></p>
</div>

From the Fig. 3 clean image above (left), the n = 1 ring can be visibly seen, but is overshadowed by the brighter pixels on the bright half, creating a partial crescent moon shaped ring mask. This is sub optimal.\
The proposed solution to this is a segmented percentile approach, where the image is segmented into quadrants and the percentile is taken locally, then recombined. This bypasses flooding or pixel overshadowing issues. When paired with a local gaussian filter on the mask pixels only, we can generate full masks for asymmetric images, as can be seen in Fig. 4 (right). 

## <brd>
**U-Net and Prediction Logic**

Since this is an image segmentation / localization model, a U-Net CNN architecture was the best model logic to use. In total 13 convolutional were used, 6 encoder and 4 decoder layers plus 1 output layer, as well as 2 convolutional transpose layers. Each layer uses a ReLU activation function and 2x2 pooling.

The ring mask prediction works using the following logic: clean simulated image -> clean ground-truth ring mask is taken -> clean image is degraded and fed into the model -> model compares the clean ground-truth masks and its corresponding degraded image -> model attempts a ring mask prediction from the degraded image -> compares predicted ring mask with the ground-truth ring mask and adjusts.\
The GRRT<sup>[3]</sup> dataset has 3 subsets; training, validation and testing. Initial training at 20 epochs yields a ~50% prediction-truth overlap and generates near-acceptable masks. This metric is not comparable to an accuracy reading, as we are measuring binary pixel values and not a pixel gradient. Although that is the goal. Below in Fig. 5 is another example of a prediction using the 20 epoch model and logic described above.

<div align="center">
  <img width="1846" height="479" alt="github temp example4" src="https://github.com/user-attachments/assets/c693cedd-87bf-4b88-a13d-15019a2a762a" />
  <p><em>Figure 5: Full prediction example 2</em></p>
</div>

Future work includes optimization of predicted ring masks, implementation of the segmented percentile ring mask algorithm mentioned, gradient ring mask predictions instead of binary masks, predictions of n = 0, 1, 2 photon ring layers and potential parameter extraction such as black hole spin, radius, etc., among other things. Of course applications of this method on real life observed black hole images is expected in the future.

(ALL INFORMATION IS SUBJECT TO CHANGE)

## <brd>
**REFERENCES**

[1]A. Kouroshnia, K. Nguyen, C. Ni, A. SaraerToosi, and A. E. Broderick, “Learning to See: Applying Inverse Recurrent Inference Machines to See through Refractive Scattering,” The Astrophysical Journal, vol. 985, no. 2, p. 200, May 2025, doi: 10.3847/1538-4357/adcabf.

[2]C. Duong, F. Myhre, and J. R. Farah, “Convolutional Neural Network for Extraction of n = 1 Photon Ring of Black Holes,” Sept. 2026, p. 21 Sep 2026. doi: arXiv:2609.25344v1.

[3]JialeiWei, “GRRT Datasets for MANet,” Zenodo, July 2025, doi: 10.5281/zenodo.15846647.

[4]J. Wei, L. Ao, D. Li, and C. Wen, “Physical parameter regression from black hole images via a multiscale adaptive neural network.” July 21, 2025. Accessed: Oct. 07, 2026. [Online]. Available: https://arxiv.org/abs/2507.15910v1
