# KAI STRAKA 2026

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import photonImage
import photonDegrade


class convBlock(nn.Module):

        def __init__(self, in_channels, out_channels):

                super().__init__()

                # conv layer block
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True),    # first conv layer
                    nn.BatchNorm2d(out_channels),   # batch channels
                    nn.ReLU(),                      # activation function

                    nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True),                                           # second conv layer
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )


        def forward(self, x):

            return self.conv(x)


class photonUNET(nn.Module):

        def __init__(self):

                super().__init__()

                # encoder layers
                self.enc1 = convBlock(1, 32)            # conv layers 1, 2
                self.enc2 = convBlock(32, 64)           # conv layers 3, 4
                self.bottleneck = convBlock(64, 128)    # conv layers 5, 6

                # decoder layers
                self.dec2 = convBlock(128, 64)          # conv layers 7, 8
                self.dec1 = convBlock(64, 32)           # conv layers 9, 10
                self.output = nn.Conv2d(32, 1, 1)       # output conv layer

                self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)             # same params as Conv2d
                self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)

                self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)       # max pooling 2x2


        def forward(self, x):

               enc1 = self.enc1(x)              # first encoder layer set
               pool1 = self.pool(enc1)          # pooling 2x2

               enc2 = self.enc2(pool1)          # second encoder layer set
               pool2 = self.pool(enc2)          # pooling 2x2

               b_neck = self.bottleneck(pool2)              # bottleneck layers

               dec2 = self.up2(b_neck)                      # first conv transpose
               dec2 = torch.cat((dec2, enc2), dim = 1)      # first skip | concatenate function
               dec2 = self.dec2(dec2)                       # first decoder layer set

               dec1 = self.up1(dec2)                        # second conv transpose
               dec1 = torch.cat((dec1, enc1), dim = 1)      # second skip
               dec1 = self.dec1(dec1)                       # second decoder layer set

               output = self.output(dec1)       # output layer set

               return output



class photonDataset(Dataset):

        def __init__(self, image_paths):

            self.image_paths = image_paths


        def __len__(self):
            
            return len(self.image_paths)


        def __getitem__(self, index):
            
            image_path = self.image_paths[index]

            deg_image = photonDegrade.degraded_image(image_path)        # load degraded image
            deg_tensor = photonImage.pixel_tensor(deg_image)            # convert degraded image to tensor

            clean_image = photonImage.pixel_list(image_path)            # load clean image

            mask = photonImage.ring_mask(clean_image)                   # clean image ring mask
            mask_tensor = photonImage.pixel_tensor(mask)                # convert clean ring mask image to tensor

            return deg_tensor, mask_tensor
