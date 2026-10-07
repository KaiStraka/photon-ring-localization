KAI STRAKA 10.2026

Neural localization of n ≥ 0 photon ring structure in synthetically degraded black hole images.

## <brd>
**WORK IN PROGRESS PROJECT**

NOTE: this README is not a detailed description or representation of the project, consider it a temporary introduction.\
Below is brief information on current and ongoing results (all information IS subject to change). 

**INTRO**

As of October 2026, this purpose of this project is using a U-Net CNN (convolutional neural network) architecture for localization and ring mask prediction of n ≥ 0 photon ring structures within simulated and synthetically degraded black hole images.\
Below is an example of the predicted ring mask pipeline in action.

<img width="1847" height="474" alt="github temp example" src="https://github.com/user-attachments/assets/74ae6a5c-8b7d-4cd6-b011-4924e2c2a0b3" />

## <brd>
**Image Degredation**

Simulated photon ring images all come from the GRRT public dataset as clean and ideal images. My model currently uses two filters to degrade the images to realistic observational conditions.

The first layer is an elliptical Gaussian blur, currently set at a standard deviation of 10 (Gauss10). This was found to provide the most realistic blurring effect.\
The second layer is an interstellar scattering screen using Kolmogorov turbulence.\
To achieve a degraded image, a clean image will first gain the Gaussian blur, then the scattering screen.\
Below is an example of the two degradation layers.

<img width="1776" height="573" alt="github temp example2" src="https://github.com/user-attachments/assets/d846c5ee-f06d-4fc6-a254-282b2aeee27d" />
