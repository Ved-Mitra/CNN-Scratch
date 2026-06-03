# CNN-Scratch (with Custom Autograd Engine)

This project is a Convolutional Neural Network (CNN) built entirely from scratch in C++. 

Instead of hardcoding the backward passes for specific layers, this project features a custom-built **Autograd Engine** (similar to PyTorch's `Tensor` system). The engine dynamically builds a computation graph during the forward pass and automatically computes gradients via reverse-mode automatic differentiation.

## Key Features
* **Custom Tensor Library:** N-dimensional array processing built from the ground up using C++ Smart Pointers.
* **Automatic Differentiation:** A PyTorch-style dynamic computation graph that tracks mathematical operations and computes gradients using the Chain Rule.
* **Hardware Abstraction:** Designed with custom memory management to allow seamless switching between CPU (`new double[]`) and GPU/CUDA (`cudaMalloc`) execution in the future.
* **Neural Network Modules:** Fully Connected (Dense) layers, Convolutional layers, Max Pooling, and Activation functions built purely on top of the Autograd tensor operations.

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
