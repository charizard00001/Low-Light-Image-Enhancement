# 🌟 Low Light Image Enhancement 🌟

## Objective 🎯
To build a deep learning model that can transform a low-light image into a high-quality, well-lit masterpiece. 📸✨

## Introduction 📚
In today's technological era, visual content plays a pivotal role in various fields such as entertainment, social media, education, and business. However, capturing high-quality images in low-light settings remains a significant challenge, often resulting in images with poor brightness, noise, and blurred details. To tackle this issue, we've developed an AI model designed to enhance low-light images.

## Overview 🔍
The project harnesses the combined power of machine learning and deep learning to achieve this task. Initially, we employ XGBoost to predict the histogram of the target image, serving as a preprocessing step for the transformer model.

## Dataset 📊
Utilized the LOL dataset to train our model. It consists of 500 images with a resolution of 400x600 pixels, out of which 485 images are used for training and 15 for evaluating the model.

## Histogram Mapping 🗺️
Through thorough data analysis, we observed a significant correlation between the pixel values of low-light and high-light images. Specifically, the pixel values of these images are closely linked in terms of percentile, i.e., the same pixel positions correspond to the same small percentile range in both images. Leveraging this insight, we employed a quantile regression model to perform histogram mapping. 

Trained an XGBoost regressor model that utilizes the low-light image's histogram as input and generates the pixel values of the high-light image at every 5th percentile. We then use these pixel values and a KDE model to reconstruct the histogram of the high-light image pixels. By applying a histogram transformation on the low-light image to the generated histogram, we successfully reconstructed a high-light version of the low-light image.

## Vision Transformer 🧠
The output image after histogram mapping still exhibited amplified noise and color distortions due to low-light settings. Therefore, we applied a vision transformer, which takes input images as patches of 10x10 and outputs new patches that, when concatenated, generate a new, enhanced image.

## Training 🏋️‍♂️
- **Loss Function:** Our objective was to minimize -PSNR + L1 Loss.
- **Epochs:** We trained for about 150 epochs, achieving a training loss of ~28.06.
- **Training Loss vs Epoch:** The training loss vs. epoch for the first 50 epochs is illustrated below.

## Results 🏆
- **Mean PSNR on the test set after histogram mapping:** 19.6
- **Mean PSNR on the test set after applying the vision transformer to histogram mapping:** 20.52

1. Install Dependancies
```bash
pip3 install -r requirements.txt
```
2. Run model on custom dataset
```bash
python3 eval.py --output <output-dir> --input <input-dir>
```
3. Training script we used to train out models
```bash
python3 train.py
```

## Conclusion 🏁
This experiments revealed that preprocessing and histogram mapping are crucial steps before applying deep learning models, as they enable the models to learn other features more effectively. This approach significantly enhances the quality of low-light images, making them more visually appealing and useful.

---

✨ **Thank you for checking out Low Light Image Enhancement!** ✨
