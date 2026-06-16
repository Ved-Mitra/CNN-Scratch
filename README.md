# CNN-Scratch (with Custom Autograd Engine)

This project is a Convolutional Neural Network (CNN) built entirely from scratch in C++. 

Instead of hardcoding the backward passes for specific layers, this project features a custom-built **Autograd Engine** (similar to PyTorch's `Tensor` system). The engine dynamically builds a computation graph during the forward pass and automatically computes gradients via reverse-mode automatic differentiation.

## Key Features
* **Custom Tensor Library:** N-dimensional array processing built from the ground up using C++ Smart Pointers.
* **Automatic Differentiation:** A PyTorch-style dynamic computation graph that tracks mathematical operations and computes gradients using the Chain Rule.
* **Neural Network Modules:** Fully Connected (Dense) layers, Convolutional layers, Max Pooling, and Activation functions (ReLU, Softmax) built purely on top of the Autograd tensor operations.

## What's Next?
* **Vectorized Mini-Batching:** Support for `[N, C, H, W]` tensor shapes to improve training stability and throughput.
* **Model Persistence:** Saving and loading trained weights to/from binary files.
* **Hardware Abstraction (CUDA):** Moving from CPU arrays to GPU acceleration via CUDA kernels.
* **Advanced Layers:** Implementing Dropout, Batch Normalization, and Leaky ReLU for better convergence.

## Architecture & Roadmap
This project is being developed in 5 core phases:
1. **Core Mathematical Foundation:** The `Tensor` class handling memory allocation, multi-dimensional shapes, and element-wise math operations.
2. **The Autograd Engine:** Wiring up the computation graph with topological sorting and `.backward()` propagation.
3. **Neural Network Layers:** Building the `Layer` interfaces (Dense, Conv2D, MaxPool) using only forward-pass math.
4. **Training Infrastructure:** Implementing Cross-Entropy Loss, Stochastic Gradient Descent (SGD) optimizer, and the training loop.
5. **Data & Utilities:** A binary data loader to parse and train the network on the MNIST dataset.

## Build Instructions
This project uses `CMake` for cross-platform building and dependency management.

```bash
mkdir build
cd build
cmake ..
make
```
## Results

### On MNIST Dataset:

training set - 50K images

test set - 10K images

classes - 10
```
--- Training on MNIST Dataset ---
Loaded 10000 samples for training.
Loaded 10000 samples for testing.
Starting MNIST Training...
Epoch 1 - Average Loss: 1.2011
Epoch 2 - Average Loss: 0.623326
Epoch 3 - Average Loss: 0.502686
Epoch 4 - Average Loss: 0.441675
Epoch 5 - Average Loss: 0.40235
Testing Model
Accuracy : 87.7%
```

### On Fashion-MNIST Dataset:

training set - 60K images

test set - 10K images

classes - 10
```
--- Testing on Fashion-MNIST Dataset ---
Loaded 2000 samples for training.
Loaded 500 samples for testing.
Starting Training...
Epoch 1 - Average Loss: 1.30685
Epoch 2 - Average Loss: 0.858823
Epoch 3 - Average Loss: 0.742627
Epoch 4 - Average Loss: 0.679142
Epoch 5 - Average Loss: 0.636513
Testing Model...
Final Accuracy: 77%
```

### On EMNIST (Balanced) Dataset:

training set - 112K images

test set - 18K images

classes - 47
```
--- Testing on EMNIST Dataset ---
Loaded 2000 samples for training.
Loaded 500 samples for testing.
Starting Training...
Epoch 1 - Average Loss: 3.03032
Epoch 2 - Average Loss: 2.00392
Epoch 3 - Average Loss: 1.67889
Epoch 4 - Average Loss: 1.51004
Epoch 5 - Average Loss: 1.39838
Epoch 6 - Average Loss: 1.31506
Epoch 7 - Average Loss: 1.24848
Epoch 8 - Average Loss: 1.19293
Epoch 9 - Average Loss: 1.14519
Epoch 10 - Average Loss: 1.1033
Testing Model...
Final Accuracy: 61%
```

### On CIFAR-10 Dataset:

training set - 50K images

test set - 10K images

classes - 10
```
--- Testing on CIFAR-10 Dataset ---
Loaded 2000 samples for training.
Loaded 500 samples for testing.
Starting Training...
Epoch 1 - Average Loss: 2.31949
Epoch 2 - Average Loss: 2.15595
Epoch 3 - Average Loss: 2.07353
Epoch 4 - Average Loss: 2.01699
Epoch 5 - Average Loss: 1.97345
Epoch 6 - Average Loss: 1.93763
Epoch 7 - Average Loss: 1.90692
Epoch 8 - Average Loss: 1.8799
Epoch 9 - Average Loss: 1.85566
Epoch 10 - Average Loss: 1.83363
Testing Model...
Final Accuracy: 29.4%
```

## Conclusion & Analysis

The testing across four distinct datasets demonstrates the versatility and correctness of the custom **Autograd Engine**. The engine successfully performed backpropagation through multiple layers (Conv2D, ReLU, MaxPool, Linear) by dynamically building the computation graph, regardless of the input's spatial dimensions or channel count.

### Performance Analysis:
1.  **MNIST (87.7%) & Fashion-MNIST (77%):** The model performs exceptionally well on these datasets. The patterns are relatively simple (grayscale, centered objects), allowing a shallow CNN to capture the essential features quickly.
2.  **EMNIST (61%):** The drop in accuracy is due to the increased complexity of **47 classes**. Discriminating between similar characters (like 'O' and '0' or 'l' and '1') requires a deeper network and significantly more training epochs to converge.
3.  **CIFAR-10 (29.4%):** This is the most challenging dataset. Unlike the previous grayscale datasets, CIFAR-10 contains **3-channel RGB images** of real-world objects in diverse backgrounds.

### Reasons for Lower Accuracy on CIFAR-10:
*   **Feature Complexity:** 3x32x32 images have much higher variance. A single 8-filter convolutional layer is insufficient to extract high-level semantic features from color images.
*   **Vanishing/Exploding Gradients:** Without **Batch Normalization**, deep signals can become unstable, making it harder for the model to learn complex color distributions.
*   **Optimization:** The current **Stochastic Gradient Descent (SGD)** updates weights sample-by-sample. This introduces a lot of noise. Implementing **Mini-Batching** would provide smoother gradient updates and better convergence.

### Future Improvements to Boost Accuracy:
*   **Mini-Batch Processing:** Vectorizing the operations to process multiple images at once.
*   **Advanced Optimizers:** Implementing **Adam** or **RMSprop** for adaptive learning rates.
*   **Batch Normalization:** To stabilize training and allow for deeper architectures.
*   **Data Augmentation:** Artificially increasing the training set size by rotating or flipping images to prevent overfitting.

