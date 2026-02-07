# Food101 Image Classification with Transfer Learning

Image classification is a core task in computer vision, enabling machines to interpret and categorise images. Convolutional Neural Networks (CNNs) are widely used for this task due to their ability to automatically learn features from raw image data, making them highly effective across applications such as healthcare, agriculture, security and autonomous vehicles.

A major challenge in image classification is the need for large labelled datsets and significant computational resources to train deep models from scratch. **Transfer learning** addresses this issue by using CNNs pre-trained on large-scaled datasets like ImageNet, allowing models to adapt to new tasks with smaller datasets and faster convergence.

This project explores **transfer learning for image classificaiton** using the **Food101 dataset** and implements experiments in **PyTorch**.

## Key objectives
* Compare three pre-trained CNN architectures: **GoogLeNet, MobileNetV3 (Large) and ResNet50.**
* Apply transfer learning and evaluate baseline performance without fine-tuning.
* Fine-tune the most promising model to improve classification accuracy.
* Analyse limitaitons observed during experiments and suggest improvements or applications.

## Dataset: Food101

The Food101 dataset contains 101,000 images spanning 101 food categories, with:
* 750 training images per class
* 250 testing images per class

The dataset is balanced and reflects real-world variability, with natural, user-generated content. This diversity makes Food101 an excellent benchmark for evaluating the generalisability of pre-trained CNNs.

Example challenges:

* Images often include distracting elements (e.g., people in the frame)
* Food items may appear in containers or partially obstructed

These factors require models to learn robust and generalisable features.

## Model Architectures Explored

| Model                   | Key Features                                                  |
| ----------------------- | ------------------------------------------------------------- |
| **GoogLeNet**           | Inception modules with multi-scale feature extraction         |
| **MobileNetV3 (Large)** | Lightweight, efficient, uses depthwise separable convolutions |
| **ResNet50**            | Residual connections for training deep networks effectively   |

## Repository Contents
```
food101/
│
├── different_architectures.ipynb  # Notebook with experiments testing different pre-trained model archiectures
├── MobileNetV3.ipynb              # Notebook with fine-tuning MobileNetV3 (final model)
├── README.md                      # Project overview and results
├── images/
    └── food101_samples.png        # Example dataset images
```

## How to Run

1. Clone the respository:

```
git clone https://github.com/Shawynot33/food101.git
```

2. Open and run notebooks:

```
jupyter notebook different_architectures.ipynb

jupyter notebook MobileNetV3.ipynb
```
