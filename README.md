
# Restoring severely out-of-focus blurred text images with DIP
## 1 - Introduction

Deep Image Prior (DIP) reconstructs images given the single degraded image and a forward model. In this work, instead of using only the DIP network, the main idea here is to use an additional supervised CNN  to include prior information from the sharp images.

**Method overview**

![result of deblured image ](https://github.com/robert-abc/HDC-DIP/blob/main/Figures/Overview.png)

## 2 - Dataset 
The dataset was made available by Finnish Inverse Problems Society (FIPS) during the Helsinki Deblur Challenge 2021 (HDC2021):

#### HDC2021 main page:
    https://www.fips.fi/HDC2021.php
    
#### Description of Photographic Data:
    http://arxiv.org/abs/2105.10233.

#### Training set
    https://zenodo.org/record/4916176

#### Test set
    https://zenodo.org/record/5713637

#### The OCR evaluation code is available at
    https://www.fips.fi/HDC2021_ocrscorecode.zip

## 3.Installation, usage instructions, and examples

### 3.1 - Prerequisites

**Python 3.7**

| Package  | Version | 
| ------------- | ------------- | 
| PyTorch  | 1.11.0  | 
| OpenCV  | 4.1.2  |  
| Pillow  | 7.1.2  | 
| Numpy | 1.21.6 | 
| Scipy | 1.4.1 | 

We recommend running the code using CUDA/GPU.

### 3.2 - Installation: Clone this repository
     git clone https://github.com/robert-abc/HDC-DIP/
     cd HDC-DIP



### 3.3 - Usage instructions
#### Training: The first command needed for the training generates the dataset of reconstructions by using only the DIP
    $ python train_weights.py path/to/blurred/files path/to/focused/files --blur_level blur_category --save_intermediary path/to/dip/results
    
#### Training: The second command train the CNN using this generated dataset as input and save the weights: 
    $ python train_weights.py path/to/blurred/files path/to/focused/files --blur_level blur_category --weight_path path/to/save/weight --have_intermediary path/to/dip/results

#### Test: The main function is a callable function from the command line:
    $ python main.py path/to/blurred/files path/to/save/deblurred/files blur_category

## 4 - Result example

The following example is an out-of-focus blur from category 15 (HDC2021). From left to right: Sharp, blurred and deblurred images.
   
![result of deblured image ](https://github.com/robert-abc/HDC-DIP/blob/main/Figures/example.png)

## 5 - Performance comparison
### We compared the results with the following algorithms

#### URL  for "Neural Blind Deconvolution Using Deep Priors ": 
    https://github.com/csdwren/SelfDeblur 

#### URL for "DeepRED: Deep Image Prior Powered by RED": 
    https://github.com/GaryMataev/DeepRED

#### URL for "Deconvblind": 
    https://www.mathworks.com/help/images/ref/deconvblind.html
