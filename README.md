# Low Light Image Enhancement

## Introduction

With the objective of restoring high-quality image content from its degraded versions, the field of image restoration has numerous applications in areas such as photography, security, medical imaging, and remote sensing. In this project, we implement the MIRNet model for low-light image enhancement. This fully-convolutional architecture is designed to learn an enriched set of features by combining contextual information from multiple scales, while simultaneously preserving high-resolution spatial details. The innovative approach of MIRNet effectively revitalizes degraded images, enhancing their clarity and quality.

## Code Explanation

### Setup

The initial code sets up the environment and imports essential libraries for the low light enhancement project. It specifies TensorFlow as the backend for Keras, ensuring compatibility and performance. Key libraries imported include:

- `os` for environment management
- `random` and `numpy` for data manipulation
- `glob` for file handling
- `PIL` for image processing
- `matplotlib.pyplot` for visualization
- `keras` and `tensorflow` for building and training the neural network model

### Data Preparation

Functions to read, preprocess, and create datasets for low light enhancement tasks:

- `read_image` reads and normalizes images.
- `random_crop` crops images to a specified size.
- `load_data` applies these operations to pairs of low light and enhanced images.
- `get_dataset` creates a TensorFlow dataset from lists of image paths, maps the load_data function to them, and batches the data.

### Model Components

- **Selective Kernel Feature Fusion**: Fuses multi-scale features using a selective kernel mechanism to enhance feature representation.
- **ChannelPooling Class**: Custom Keras layer performing channel pooling by combining average and max pooling operations.
- **Attention Blocks**: 
  - `spatial_attention_block` emphasizes spatially significant regions.
  - `channel_attention_block` emphasizes important channels.
  - `dual_attention_unit_block` integrates both spatial and channel attention mechanisms.
- **Downsampling and Upsampling Modules**:
  - `down_sampling_module` reduces spatial dimensions and increases channel dimensions.
  - `up_sampling_module` increases spatial dimensions and decreases channel dimensions.
- **Multi-Scale Residual Blocks**:
  - `multi_scale_residual_block` integrates downsampling, attention mechanisms, and feature fusion.
  - `recursive_residual_group` creates a group of multi-scale residual blocks.
- **MIRNet Model**: The overall model structure using recursive residual groups and multi-scale residual blocks.

### PSNR Calculation

The PSNR values calculated for each image, along with the average PSNR across all images, indicate the model's performance in enhancing the low-light images.

## Results

The PSNR values for the test images are as follows:

- PSNR for image 1: 27.70 dB
- PSNR for image 2: 27.74 dB
- PSNR for image 3: 27.34 dB
- PSNR for image 4: 27.51 dB
- PSNR for image 5: 28.30 dB
- PSNR for image 6: 27.82 dB
- PSNR for image 7: 28.24 dB
- PSNR for image 8: 27.42 dB
- PSNR for image 9: 27.85 dB
- PSNR for image 10: 27.78 dB
- PSNR for image 11: 27.38 dB
- PSNR for image 12: 27.52 dB
- PSNR for image 13: 27.59 dB
- PSNR for image 14: 27.95 dB
- PSNR for image 15: 28.62 dB

**Average PSNR: 27.79 dB**

These PSNR values indicate significantly higher quality of enhancement, with an average PSNR of approximately 27.79 dB across all images.

## Conclusion

This project successfully implemented a custom image enhancement model for low-light images and evaluated its performance using PSNR (Peak Signal-to-Noise Ratio). The script automated the process of enhancing images, calculating PSNR, and saving results, providing quantitative insights into image quality improvements.

## Future Recommendations

Moving forward, consider exploring:

- Advanced enhancement techniques like GANs or CNNs.
- Integrating additional evaluation metrics beyond PSNR for comprehensive assessment.
- Enhancing dataset diversity.
- Optimizing for real-time processing.
- Developing user-friendly interfaces.
- Continuously refining performance for broader applicability and usability.

