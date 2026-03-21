# Deep Dreaming using Convolutional Neural Networks

This project was developed as part of the Machine Learning course at Hochschule Fulda. It focuses on implementing the Deep Dream technique to visualize and understand the internal representations learned by deep neural networks.

The project demonstrates how neural networks can be used not only for prediction but also for generating and transforming images.

---

## Academic Context

This project is based on the concept of deep dreaming, a technique introduced by DeepMind to visualize what neural networks learn.

The goal is to modify an input image in such a way that it maximizes the activations of a selected layer in a pre-trained neural network.

---

## Project Objective

The main objectives of this project are:

- Understand internal feature representations of deep neural networks
- Implement gradient-based image optimization
- Use pre-trained models for feature visualization
- Generate transformed images using Deep Dream

---

## Concept Overview

Deep Dream works by:

- Taking a trained neural network
- Keeping model weights fixed
- Modifying the input image
- Maximizing activations of a chosen layer using gradient ascent

This answers the question:

"What input image maximizes the activation of a given layer?"

---

## Methodology

### Model Selection

- Pre-trained MobileNetV2 model (ImageNet)
- Fully connected layers removed
- Intermediate layers used for feature extraction

---

### Image Processing

- Input image loaded using PIL
- Resized to 224 x 224 x 3
- Normalized to range [0, 1]

---

### Deep Dream Optimization

The algorithm performs:

- Forward pass through selected layer
- Loss defined as sum of squared activations
- Gradient computed with respect to input image
- Image updated using gradient ascent

Implemented using:

- TensorFlow GradientTape for gradient computation
- Manual update of image pixels

---

### Multi-Scale Processing (Octaves)

To enhance visual quality:

- Image is progressively upscaled
- Deep dreaming applied at multiple resolutions
- Improves fine details and patterns

---

### Output Generation

- Intermediate images saved every few iterations
- Final transformed image saved as:
  - dream_<image_name>.png

---

## Implementation Details

The script dream.py includes:

- Dynamic selection of network layer
- Gradient normalization for stable updates
- Clipping pixel values to valid range
- Iterative optimization process

Command-line usage:

python3 dream.py <image> <layer_index> <iterations>

---

## Key Observations

- Different layers produce different visual patterns
- Lower layers enhance textures and edges
- Higher layers generate complex and abstract features
- Output strongly depends on the selected layer

From the experiments:

- Simple images evolve into highly abstract, pattern-rich visuals
- Neural networks amplify features they have learned during training

---

## Conclusion

This project demonstrates an alternative use of neural networks beyond classification.

Key conclusions:

- Neural networks encode rich hierarchical features
- Gradient-based optimization can reveal internal representations
- Deep Dream provides insight into what networks focus on
- Model interpretability is an important aspect of machine learning

---

## Learning Outcomes

Through this project, the following concepts were applied:

- Understanding convolutional neural networks at a deeper level
- Gradient-based optimization techniques
- Use of pre-trained models for feature extraction
- Model interpretability and visualization
- TensorFlow and Keras advanced usage

---

## Project Structure

├── dream.py
├── input_images/
├── outputs/
└── README.md

---

## How to Run

1. Install dependencies:

pip install tensorflow pillow matplotlib numpy

2. Run the script:

python3 dream.py input.jpg 50 100

- 50 refers to the selected layer index
- 100 is the number of gradient ascent steps

---

## Author

Pavithra Lakshmi Venugopal  
M.Sc. Data Science  
Hochschule Fulda
