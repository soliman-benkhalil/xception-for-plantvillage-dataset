# Plant Disease Classification using Deep Learning 🌿

## Project Overview
This project implements and compares three state-of-the-art deep learning architectures (ResNet-18, DenseNet, and Xception) for plant disease classification using the PlantVillage dataset. The comparison demonstrates Xception's superior performance in terms of accuracy and generalization.

## Dataset Information
- **Source**: PlantVillage Dataset
- **Total Images**: 54,305
- **Number of Classes**: 38
- **Test Set Size**: 5,431 samples
- **Classes Include**: Various plant diseases across different species including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato

## Model Architecture Comparison

### Xception (Best Performing)
- **Test Accuracy**: 98%
- **Macro Average F1-Score**: 0.97
- **Key Features**:
  - Depthwise separable convolutions
  - Enhanced feature extraction
  - Excellent generalization capabilities
  - No overfitting observed

### Performance Highlights
- Perfect accuracy (1.00) achieved for multiple classes including:
  - Apple (healthy)
  - Blueberry (healthy)
  - Orange (Citrus greening)
  - Soybean (healthy)
  - Squash (Powdery mildew)
  - Strawberry (healthy)

### Notable Class-wise Performance
```
Class                                    Precision  Recall  F1-Score
---------------------------------------- ---------- ------- --------
Apple___Apple_scab                       0.97       0.95    0.96
Tomato___Early_blight                    0.92       0.79    0.85
Grape___Black_rot                        0.97       0.98    0.98
Orange___Haunglongbing                   1.00       1.00    1.00
```

## Implementation Details

### Data Processing
- Stratified data splitting
- Image augmentation techniques
- Normalized input preprocessing
- Balanced class distribution

### Training Strategy
- Learning rate scheduling
- Early stopping mechanism
- Dropout regularization
- Cross-validation implementation
- Batch normalization

### Model Optimization
- Dynamic learning rate adjustment
- Gradient clipping
- Weight decay regularization
- Batch size optimization

## Results Analysis

### Comparative Metrics
1. **Xception**:
   - Best overall performance
   - Superior generalization
   - No overfitting observed
   - Highest macro F1-score

2. **ResNet-18**:
   - Good performance
   - Slightly lower generalization
   - Competitive accuracy

3. **DenseNet**:
   - Comparable performance
   - Higher parameter count
   - Longer training time

### Key Achievements
- 98% overall accuracy on test set
- 0.97 macro-average F1-score
- Robust performance across diverse plant species
- Excellent handling of class imbalance

## Usage Instructions

### Requirements
```
python>=3.8
torch>=1.8.0
torchvision>=0.9.0
pandas
numpy
scikit-learn
matplotlib
```

