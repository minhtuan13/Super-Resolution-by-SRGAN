# Super Resolution by SRGAN

This project implements a Super-Resolution Generative Adversarial Network (SRGAN), designed to upscale low-resolution images to high-resolution images with perceptual quality. SRGAN combines a Generator and a Discriminator in an adversarial setting, where the Generator learns to create realistic high-resolution images and the Discriminator learns to distinguish between real and fake images.

# Features
- Super-resolution using GAN architecture (SRGAN).
- Perceptual Loss: Uses a pre-trained VGG19 network to calculate perceptual similarity between images.
- Adversarial Loss: Encourages the Generator to produce more realistic images using feedback from the - Discriminator.
- Customizable configurations for training.
- Checkpoint saving and loading for continuing training.
- Easy-to-use test and evaluation functions to generate high-resolution images.

# Training
- Data Preparation: Prepare a dataset of low-resolution and high-resolution image pairs. Update config.py with the paths to your training and validation datasets.

- Configure Training Settings: Modify hyperparameters such as batch size, learning rate, and number of epochs in config.py.

- Run Training: Use the command to start training. The model will save checkpoints during the process, allowing you to resume training at any point.

# Evaluation
- After training, you can evaluate the model’s performance by generating super-resolved images from a test dataset.
- Generated images will be saved in the specified output directory.
