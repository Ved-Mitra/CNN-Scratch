#include <iostream>
#include "model.h"
#include "dataloader.h"

int main() {
    try {
        std::cout << "--- Training on MNIST Dataset ---" << std::endl;

        // 1. Load Data
        std::string train_images_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/MNIST/train-images.idx3-ubyte";
        std::string train_labels_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/MNIST/train-labels.idx1-ubyte";

        //test data
        std::string test_images_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/MNIST/t10k-images.idx3-ubyte";
        std::string test_labels_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/MNIST/t10k-labels.idx1-ubyte";

        auto train_images = DataLoader::load_mnist_images(train_images_path);
        auto train_labels = DataLoader::load_mnist_labels(train_labels_path);

        auto test_images = DataLoader::load_mnist_images(test_images_path);
        auto test_labels=DataLoader::load_mnist_labels(test_labels_path);

        std::cout << "Loaded " << train_images.size() << " samples for training." << std::endl;
        std::cout << "Loaded" << test_images.size() << " samples for testing." << std::endl;

        // 2. Define Model Architecture for MNIST
        // Input: 784 (28x28 flattened)
        // 2. Define CNN Architecture for MNIST
        // Input: 1x28x28
        // Conv1: 1 -> 8 filters (3x3), stride=2, padding=1
        // Output after Conv1: 8 x 14 x 14
        // MaxPool: 2x2, stride=2 -> 8 x 7 x 7
        // Flatten: 8 * 7 * 7 = 392
        // Output: 10
        SimpleModel model(0.01, 10); 
        model.conv2d(1, 8, 3, 2, 1);
        model.relu();
        model.maxpool2d(2, 2, 0);
        model.flatten();
        model.linearlayer(392, 10);

        // 3. Training
        // Since our current model.train() takes total Tensors, we need to handle 
        // the batch or perform a simple loop here if your train() isn't vectorized.
        // For now, let's train on a meaningful subset to see if it learns.
        
        std::cout << "Starting MNIST Training..." << std::endl;
        int epochs = 10; 
        int num_samples = 100; // Subset for testing
        model.fit(train_images,train_labels,epochs,num_samples);
    
        std::cout << "Testing Model" << std::endl;
        auto pred=model.test(test_images);
        double accuracy=model.accuracy_score(pred,test_labels);
        std::cout << "Accuracy : " << accuracy << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        std::cerr << "Ensure dataset paths are correct relative to the execution directory." << std::endl;
    }

    return 0;
}
