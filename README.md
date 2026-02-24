# Deep Learning From Scratch: MLP, Autoencoder & RBM with NumPy

A comprehensive implementation of fundamental deep learning architectures built from scratch using Python and NumPy, without relying on high-level deep learning frameworks. This project demonstrates the core principles of neural networks through implementations of Multi-Layer Perceptron (MLP), Sparse Autoencoders, and Restricted Boltzmann Machines (RBM) trained on the Fashion-MNIST dataset.

## Overview

This project implements three key deep learning models from first principles:

1. **2-Layer Multi-Layer Perceptron (MLP)**: A feedforward neural network for image classification with backpropagation training
2. **Sparse Autoencoder**: An unsupervised learning model for feature extraction and dimensionality reduction with sparsity constraints
3. **Restricted Boltzmann Machine (RBM)**: A probabilistic graphical model for representation learning and generative modeling

## Features

- **Pure NumPy Implementation**: All models built using only NumPy, providing educational insights into mathematical foundations
- **Fashion-MNIST Dataset**: Training and evaluation on 70,000 fashion item images (28x28 pixels, 10 classes)
- **Complete Pipeline**: Data loading, preprocessing, normalization, model training, and evaluation
- **Activation Functions**: ReLU, Sigmoid, Tanh, and Softmax implementations with derivatives
- **Loss Functions**: Cross-entropy loss for classification and reconstruction loss for autoencoders
- **Visualization**: Matplotlib-based visualization of reconstructions and learned representations

## Project Structure

```
├── Fashion_MNIST_NN.ipynb    # Main Jupyter notebook with all implementations
└── README.md                 # Project documentation
```

## Models Implemented

### 1. Two-Layer MLP
- **Input Layer**: 784 neurons (28×28 flattened images)
- **Hidden Layer**: Configurable size with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation
- **Training**: Backpropagation with gradient descent
- **Purpose**: Image classification of fashion items

### 2. Sparse Autoencoder
- **Encoder**: Compresses input to hidden representation with sparsity constraint
- **Decoder**: Reconstructs original input from hidden representation
- **Sparsity**: KL-divergence penalty to encourage sparse hidden activations
- **Purpose**: Unsupervised feature learning and data reconstruction

### 3. Restricted Boltzmann Machine (RBM)
- **Visible Units**: Binary representation of input images
- **Hidden Units**: Configurable number of feature detectors
- **Training**: Contrastive Divergence algorithm
- **Purpose**: Generative modeling and representation learning

## Dependencies

- Python 3.6+
- NumPy: Numerical computing and array operations
- TensorFlow/Keras: For Fashion-MNIST dataset loading
- Matplotlib: Data visualization

Install dependencies:
```bash
pip install numpy tensorflow matplotlib
```

## Dataset

**Fashion-MNIST**: A dataset of 70,000 grayscale images (28×28 pixels) of 10 fashion item categories:
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

Training Set: 60,000 images
Validation Set: 10,000 images

## Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Fashion_MNIST_NN.ipynb
   ```

2. **Run Cells Sequentially**: Execute cells from top to bottom to:
   - Load and preprocess the Fashion-MNIST dataset
   - Initialize and train the MLP classifier
   - Initialize and train the Sparse Autoencoder
   - Initialize and train the RBM
   - Visualize results and compare model performance

## Key Implementation Details

### Activation Functions
- **ReLU**: $f(x) = \max(0, x)$ - Used in hidden layers for non-linearity
- **Softmax**: Converts logits to probability distribution for multi-class classification
- **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$ - Used in RBM
- **Tanh**: Hyperbolic tangent function for alternative non-linearity

### Loss Functions
- **Cross-Entropy Loss**: For classification tasks
- **Mean Squared Error (MSE)**: For reconstruction and regression tasks

### Optimization
- **Gradient Descent**: Parameter updates based on computed gradients
- **Configurable Learning Rate**: Control step size during training
- **Batch Processing**: Support for mini-batch training

## Results

The notebook includes:
- Classification accuracy metrics on the validation set
- Reconstructed image visualization from autoencoders
- Learned feature filters from RBM
- Training loss curves and convergence analysis
- Comparison of model architectures and performance

## Learning Outcomes

By studying this implementation, you will understand:
- How neural networks learn through backpropagation
- Mathematical foundations of gradient descent and optimization
- Unsupervised learning with autoencoders
- Probabilistic models and energy-based learning
- Trade-offs between model complexity and interpretability

## Author

Implementation by: Jitheshkumar

## License

This project is open source and available under the MIT License.

## Contributing

Contributions, improvements, and bug reports are welcome! Feel free to fork and submit pull requests.

## References

- LeCun, Y., Bengio, Y., & Hinton, G. E. (2015). "Deep learning" - Nature
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - MIT Press
- Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks"
