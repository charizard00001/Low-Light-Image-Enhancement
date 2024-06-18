import os
import random
import numpy as np # type: ignore
from glob import glob
from PIL import Image, ImageOps # type: ignore
import matplotlib.pyplot as plt # type: ignore
import keras # type: ignore
from keras import layers # type: ignore
import tensorflow as tf # type: ignore
from load_data import read_image, random_crop, load_data, get_dataset
from model import recursive_residual_group, mirnet_model
from loss import charbonnier_loss, peak_signal_noise_ratio
from infer import infer

random.seed(10)

IMAGE_SIZE = 128
BATCH_SIZE = 32
MAX_TRAIN_IMAGES = 300

train_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[:MAX_TRAIN_IMAGES]
train_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))[:MAX_TRAIN_IMAGES]

val_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[MAX_TRAIN_IMAGES:]
val_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))[MAX_TRAIN_IMAGES:]

test_low_light_images = sorted(glob("./lol_dataset/eval15/low/*"))
test_enhanced_images = sorted(glob("./lol_dataset/eval15/high/*"))


train_dataset = get_dataset(train_low_light_images, train_enhanced_images, BATCH_SIZE)
val_dataset = get_dataset(val_low_light_images, val_enhanced_images, BATCH_SIZE)

print("Train Dataset:", train_dataset.element_spec)
print("Val Dataset:", val_dataset.element_spec)
 

# for training the model
# model = mirnet_model(num_rrg=3, num_mrb=2, channels=64)

# #model.summary()

# optimizer = keras.optimizers.Adam(learning_rate=1e-4)
# model.compile(optimizer=optimizer,loss=charbonnier_loss,metrics=[peak_signal_noise_ratio],)

# history = model.fit(train_dataset,validation_data=val_dataset,epochs=50,callbacks=[keras.callbacks.ReduceLROnPlateau(monitor="val_peak_signal_noise_ratio",factor=0.5,patience=5,verbose=1,min_delta=1e-7,mode="max",)],)

# model_save_path = './trained_model.keras'
# model.save(model_save_path)

# Load the trained model
model_save_path = './trained_model.keras'
model = keras.models.load_model(model_save_path, custom_objects={
    'charbonnier_loss': charbonnier_loss,
    'peak_signal_noise_ratio': peak_signal_noise_ratio
})

# Directory paths
input_dir = './lol_dataset/eval15/high'  # Directory containing low-light images
output_dir = './test/predicted'  # Directory to save enhanced images
os.makedirs(output_dir, exist_ok=True)

# List of test low-light image paths
test_low_light_images = os.listdir(input_dir)
test_low_light_images = [os.path.join(input_dir, img) for img in test_low_light_images]

# Define the target size (should match the input size of your model)
IMAGE_SIZE = 128
target_size = (IMAGE_SIZE, IMAGE_SIZE)

# Initialize lists to store PSNR values
psnr_values = []

# Loop through test images, predict, save enhanced images, and calculate PSNR
for i, low_light_image_path in enumerate(test_low_light_images):
    # Load and preprocess original low-light image
    original_image = Image.open(low_light_image_path).resize(target_size)
    
    # Predict enhanced image
    enhanced_image = infer(original_image, model)
    
    # Save the enhanced image
    enhanced_image_path = os.path.join(output_dir, f"enhanced_image_{i+1}.jpg")
    enhanced_image.save(enhanced_image_path)
    print(f"Saved enhanced image {i+1} to {enhanced_image_path}")
    
    # Calculate PSNR between original and enhanced images
    original_np = np.array(original_image)
    enhanced_np = np.array(enhanced_image)
    psnr_value = tf.image.psnr(original_np, enhanced_np, max_val=255.0)
    psnr_values.append(psnr_value.numpy())

    # Print PSNR value
    print(f"PSNR for image {i+1}: {psnr_value.numpy()} dB")

# Calculate average PSNR
average_psnr = np.mean(psnr_values)
print(f"Average PSNR: {average_psnr} dB")