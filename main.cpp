#include <iostream>
#include "model.h"
#include "dataloader.h"

int main() {
    try {
        std::cout << "--- Training on MNIST Dataset ---" << std::endl;

        // 1. Load Data
        std::string train_images_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/MNIST/train-images.idx3-ubyte";
        std::string train_labels_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/MNIST/train-labels.idx1-ubyte";

        auto images = DataLoader::load_mnist_images(train_images_path);
        auto labels = DataLoader::load_mnist_labels(train_labels_path);

        std::cout << "Loaded " << images.size() << " samples for training." << std::endl;

        // 2. Define Model Architecture for MNIST
        // Input: 784 (28x28 flattened)
        // 2. Define CNN Architecture for MNIST
        // Input: 1x28x28
        // Conv1: 1 -> 8 filters (3x3), stride=2, padding=1
        // Output after Conv1: 8 x 14 x 14 (if calculated correctly)
        // Flatten: 8 * 14 * 14 = 1568
        // Hidden: 10
        SimpleModel model(0.01, 10); 
        model.conv2d(1, 8, 3, 2, 1);
        model.relu();
        model.flatten();
        model.linearlayer(1568, 10);

        // 3. Training
        // Since our current model.train() takes total Tensors, we need to handle 
        // the batch or perform a simple loop here if your train() isn't vectorized.
        // For now, let's train on a meaningful subset to see if it learns.
        
        std::cout << "Starting MNIST Training..." << std::endl;
        int epochs = 10; 
        int num_samples = 100; // Subset for testing

        for(int e = 0; e < epochs; e++) {
            double total_epoch_loss = 0;
            for(int i = 0; i < num_samples; i++) {
                total_epoch_loss += model.train(images[i], labels[i]);
            }
            std::cout << "Epoch " << e + 1 << " - Average Loss: " << (total_epoch_loss / num_samples) << std::endl;
        }
        
        std::cout << "\n--- Training Complete ---" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        std::cerr << "Ensure dataset paths are correct relative to the execution directory." << std::endl;
    }

    return 0;
}
