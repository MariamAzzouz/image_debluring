# Image Deblurring with Deep Learning

A deep learning model to restore sharp images from blurred ones using a custom U-Net architecture with residual blocks and perceptual loss.

## 🎯 Overview

This project implements an end-to-end solution for image deblurring using deep learning. The model is trained on the GoPro dataset and uses a combination of U-Net architecture, residual blocks, and perceptual loss to achieve high-quality deblurring results.

## 🏗️ Architecture

The model architecture consists of:
- **Encoder-Decoder Network**: Modified U-Net architecture
- **Residual Blocks**: For better feature preservation
- **Skip Connections**: To maintain fine details
- **VGG-based Perceptual Loss**: For improved visual quality

### Model Components

1. **Encoder Path**:
   - 4 encoder blocks with increasing filters (64, 128, 256, 512)
   - Each block contains double convolution and max pooling
   - Batch normalization and ReLU activation

2. **Bridge**:
   - 2 residual blocks with 512 filters
   - Maintains spatial information

3. **Decoder Path**:
   - 4 decoder blocks with decreasing filters (512, 256, 128, 64)
   - Transpose convolution for upsampling
   - Skip connections from encoder
   - Batch normalization and ReLU activation

## 💡 Key Features

### Data Preprocessing
- Image resizing to 256x256
- Normalization to [0,1] range
- Train/validation/test split (70/10/20)
- Data augmentation using TensorFlow Dataset API

### Custom Loss Function
The model uses a combination of three losses:
total_loss = 0.5 MSE + 0.3 MAE + 0.2 perceptual_loss

### Training Features
- Batch size: 16
- Learning rate: 1e-4
- Early stopping
- Learning rate reduction on plateau
- Model checkpointing

## 📊 Classes and Methods

### DataPreprocessor
- `__init__(base_path, target_size)`: Initialize preprocessor
- `load_and_preprocess_image(image_path)`: Load and normalize images
- `create_dataset(blurred_dir, sharp_dir)`: Create paired dataset
- `split_dataset(test_size, validation_size)`: Split into train/val/test
- `visualize_samples(num_samples)`: Visualize random samples

### DeblurModel
- `residual_block(x, filters)`: Create residual blocks
- `encoder_block(x, filters)`: Create encoder blocks
- `decoder_block(x, skip, filters)`: Create decoder blocks
- `build_model()`: Construct the full model
- `perceptual_loss()`: Calculate VGG-based loss
- `custom_loss()`: Combined loss function

### TrainingConfig
- Batch size configuration
- Learning rate settings
- Checkpoint paths
- Training epochs

## 📈 Results

The model is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Perceptual Loss
- Visual comparison of deblurred images

## 🙏 Acknowledgments

- GoPro dataset for training data
- TensorFlow team for the framework
- VGG19 model for perceptual loss
