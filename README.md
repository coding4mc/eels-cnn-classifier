# EELS Data Analysis and CNN-based Classification

This repository contains a machine learning framework for analyzing and classifying Electron Energy Loss Spectroscopy (EELS) data using Convolutional Neural Networks (CNNs).

## Overview

The project focuses on processing, analyzing, and classifying EELS spectra data, particularly for nickel (Ni) oxidation state classification (Ni 2+ vs Ni 3+). It implements various CNN architectures to perform spectral analysis and classification of EELS data obtained from different samples.

## Key Components

### Data Processing

- **Process Test Data**: Converts raw EELS data (in HyperSpy format) to normalized, properly scaled data suitable for machine learning
- **Data Augmentation**: Enhances training data through techniques like:
  - Shifting spectra along the energy axis
  - Adding controlled noise to spectra
  - Rescaling spectra resolution

### Machine Learning Models

The repository implements several CNN architectures with different features:

- **CNN**: Basic CNN architecture with three convolutional layers, max pooling, and dropout regularization
- **CNN2**: Enhanced model with dilated convolutions to increase receptive field and global average pooling
- **CNN3**: Advanced model with a Spatial Transformer Network (STN) for automatic alignment of spectra
- **CNN4**: Includes batch normalization, dilated convolutions, and data augmentation during training
- **CNN5**: Sophisticated architecture with attention mechanisms, batch normalization, and dilated convolutions
- **CNN6**: Most advanced model featuring residual blocks, squeeze-and-excitation attention, and LeakyReLU activations

#### CNN Architecture Comparison

| Model | Key Features | Best For |
|-------|-------------|----------|
| CNN   | Basic 3-layer CNN with dropout | Baseline performance, simpler spectra |
| CNN2  | Dilated convolutions, global avg pooling | Better feature extraction at different scales |
| CNN3  | Spatial Transformer Network | Spectra with alignment/shift variations |
| CNN4  | Batch normalization, data augmentation | Improved training stability |
| CNN5  | Attention mechanisms, dilated convolutions | Complex spectral features, noise resistance |
| CNN6  | Residual blocks, SE attention | Highest accuracy, most complex spectra |

### Training and Prediction

- **Model Training**: Handles the training process with validation metrics and model saving
- **Model Execution**: Applies trained models to test data to generate predictions
- **Visualization**: Provides tools for visualizing spectra, predictions, and probabilities

## Data Structure

- **Training Data**: Labeled spectra with known oxidation states
- **Test Data**: Unlabeled spectra from various samples (FIB, VCP, Pristine)
- **Processed Test Data**: Test data after preprocessing
- **Trained CNN Model**: Saved model weights after training

## Workflow

1. **Data Processing**: Raw EELS data is processed using `process-test-data.py` to normalize and scale the spectra
2. **Data Augmentation**: Training data is enhanced using `augment-data.py` to improve model robustness
3. **Model Training**: CNNs are trained using `train_cnn.py` with the processed training data
4. **Prediction**: Trained models are applied to test data using `cnn_predict.py` to classify oxidation states
5. **Visualization**: Results are visualized as heatmaps showing predicted classes and probabilities

## Technical Details

- **Framework**: PyTorch for neural network implementation
- **Data Format**: HyperSpy for EELS data handling
- **Hardware Acceleration**: Uses Metal Performance Shaders (MPS) for GPU acceleration on macOS

## Development Environment Setup

### Prerequisites

- Python 3.9+ (recommended: Python 3.13)
- PyTorch 2.0+
- HyperSpy 2.0+
- NumPy, Matplotlib, scikit-learn

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install torch numpy matplotlib scikit-learn hyperspy exspy
   ```

## Key Files

- `augment-data.py`: Enhances training data through augmentation techniques
- `process-test-data.py`: Prepares the test data (the data that the ML model will be applied on)
- `train_cnn.py`: Trains CNN models on the prepared data
- `cnn_predict.py`: Main script for applying trained models to test data
- `l3-l2-ratios.py`: Calculates the L3/L2 ratios of a spectra
- `lib/ml/*.py`: CNN model architectures and training utilities
- `lib/model/*.py`: Data structures for training and test data
- `lib/func/*.py`: Utility functions for data processing and loading
- `lib/plot/*.py`: Visualization tools for spectra and results

## Scientific Context

This project is designed for materials science research, specifically for analyzing the oxidation states of nickel in various samples. The CNN-based approach allows for automated classification of EELS spectra, which can reveal information about the chemical state and electronic structure of materials at the nanoscale.
