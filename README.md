# Intro to AI: Denoising Autoencoder Project

This project trains an autoencoder to remove synthetic noise/text from generated images.

## File Overview

### data preparation

- **dataset.py**: Defines `TrainingDataset` and `TestDataset` classes to load and pair noisy and original images.
- **data.py**: Provides helper functions for image loading and preprocessing transforms.
- **gen_data.py**: Generates synthetic images with optional text/noise for training and testing.

### model

- **train.py**: Generates data, trains the autoencoder, and evaluates results end-to-end.
- **AutoEncoder.py**: Implements the encoder-decoder architecture of the denoising autoencoder.
- **model.py**: generates the model for detected form lines refinement.

### main

- **main.py**: Main entry point for the denoising autoencoder project
- **recover_outline.py**: Applies the trained model to recover image outlines and remove noise.
