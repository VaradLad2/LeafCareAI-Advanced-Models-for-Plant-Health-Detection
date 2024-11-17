
# LeafHealthNet: Diverse Approaches for Plant Health Detection

This repository explores various independent approaches for detecting the health of plant leaves using custom machine learning and deep learning models. Built on a custom dataset of four leaf types, this project integrates state-of-the-art models and aims to evolve into an interactive chatbot for plant health analysis.

## Models Implemented
- **CNN (VGG16)**: A convolutional neural network based on the VGG16 architecture, demonstrating effective feature extraction and classification.
- **MobileNetV2**: A lightweight, pre-trained architecture optimized for real-time health detection.
- **CNN with Hyperparameter Tuning**: A VGG16-based CNN fine-tuned with advanced hyperparameter optimization for enhanced performance.
- **CNN+LSTM**: A hybrid model that combines CNN for feature extraction and LSTM for analyzing textual comments generated using GPT-3.

## Custom Dataset
This project uses a custom dataset consisting of images of four leaf types, each categorized as healthy or rotten. The dataset is uniquely organized to create eight distinct classes:
1. **Cheeku (Healthy/Rotten)**
2. **Guava (Healthy/Rotten)**
3. **Custard Apple (Healthy/Rotten)**
4. **Grapes (Healthy/Rotten)**

### Dataset Versions:
- **Leaf Project File**: Organizes the dataset into eight classes based on leaf type and health status.
- **Alternative Organization**: Another version with the same content, structured differently, where healthy and rotten images are stored separately but share similar content.

### Dataset Summary:
- Total Classes: **8**
- Preprocessed Image Size: **224x224**
- Augmentations: Rotation, flipping, zoom, and shear transformations.

## Features
- **Dataset Preprocessing**:
  - Augmentation techniques for improved generalization.
  - Resizing and normalization for consistency.
- **Model Training**:
  - Modular, customizable Colab notebooks for each model.
  - Use of learning rate schedulers and dropout layers for optimization.
- **Text Analysis**:
  - Textual comments generated and processed using GPT-3 to add an additional dimension to the classification task.
- **Evaluation**:
  - Comprehensive metrics and visualization of training/validation accuracy and loss.

## Future Plans
- **Interactive Chatbot**:
  - Extend the models to an interactive chatbot capable of real-time plant health detection and providing actionable insights.

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/leafhealthnet.git
cd leafhealthnet
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Preprocess Dataset
Organize and preprocess the dataset using:
```bash
python preprocess.py
```

### 2. Train Models
Run the respective Colab notebooks for training:
- `cnn_vgg16.h5` (Standard CNN model using VGG16)
- `mobilenetv2_training.h5`
- `cnn_vgg16_hyperparameter_tuning.h5`
- `cnn_lstm_combined.h5`

### 3. Evaluate Models
Evaluate trained models using:
```bash
python evaluate.py
```

## Results
- **CNN (VGG16)**: Robust baseline model with high accuracy.
- **MobileNetV2**: Lightweight architecture suitable for real-time inference.
- **CNN with Hyperparameter Tuning**: Achieved improved accuracy and generalization with parameter optimization.
- **CNN+LSTM**: Combined visual and textual analysis for comprehensive classification.

## Contributions
This project is an independent effort, with comments generated and processed using GPT-3. Contributions and suggestions are welcome for enhancing models or developing the chatbot.

## License
This project does not currently have a license. All rights are reserved by the author.

---

