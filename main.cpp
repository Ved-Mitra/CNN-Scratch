#include <iostream>
#include "model.h"
#include "dataloader.h"
int main() {
    try {
        std::cout << "--- CNN Scratch Testing Environment ---" << std::endl;

        std::vector<std::shared_ptr<Tensor>> train_images, train_labels, test_images, test_labels;
        int num_classes, img_rows, img_cols, img_channels;
        int train_samples = 2000;
        int test_samples = 500;
        int epochs = 10;
        double learning_rate = 0.005;

        // --- 1. MNIST ---
        /*
        std::cout << "Dataset: MNIST" << std::endl;
        num_classes = 10; img_rows = 28; img_cols = 28; img_channels = 1;
        std::string base_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/MNIST/";
        train_images = DataLoader::load_idx_images(base_path + "train-images.idx3-ubyte", train_samples);
        train_labels = DataLoader::load_idx_labels(base_path + "train-labels.idx1-ubyte", num_classes, train_samples);
        test_images = DataLoader::load_idx_images(base_path + "t10k-images.idx3-ubyte", test_samples);
        test_labels = DataLoader::load_idx_labels(base_path + "t10k-labels.idx1-ubyte", num_classes, test_samples);
        */

        // --- 2. Fashion-MNIST ---
        /*
        std::cout << "Dataset: Fashion-MNIST" << std::endl;
        num_classes = 10; img_rows = 28; img_cols = 28; img_channels = 1;
        std::string base_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/FashionMNIST/";
        train_images = DataLoader::load_idx_images(base_path + "train-images-idx3-ubyte", train_samples);
        train_labels = DataLoader::load_idx_labels(base_path + "train-labels-idx1-ubyte", num_classes, train_samples);
        test_images = DataLoader::load_idx_images(base_path + "t10k-images-idx3-ubyte", test_samples);
        test_labels = DataLoader::load_idx_labels(base_path + "t10k-labels-idx1-ubyte", num_classes, test_samples);
        */

        // --- 3. EMNIST (Balanced) ---
        /*
        std::cout << "Dataset: EMNIST (Balanced)" << std::endl;
        num_classes = 47; img_rows = 28; img_cols = 28; img_channels = 1;
        std::string base_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/EMINST/";
        train_images = DataLoader::load_idx_images(base_path + "emnist-balanced-train-images-idx3-ubyte", train_samples);
        train_labels = DataLoader::load_idx_labels(base_path + "emnist-balanced-train-labels-idx1-ubyte", num_classes, train_samples);
        test_images = DataLoader::load_idx_images(base_path + "emnist-balanced-test-images-idx3-ubyte", test_samples);
        test_labels = DataLoader::load_idx_labels(base_path + "emnist-balanced-test-labels-idx1-ubyte", num_classes, test_samples);
        */

        // --- 4. CIFAR-10 ---
        std::cout << "Dataset: CIFAR-10" << std::endl;
        num_classes = 10; img_rows = 32; img_cols = 32; img_channels = 3;
        std::string base_path = "/home/ved/Desktop/PROJECTS/CNN Scratch/dataset/CIFAR_10/";
        DataLoader::load_cifar_binary(base_path + "data_batch_1.bin", train_images, train_labels, train_samples);
        DataLoader::load_cifar_binary(base_path + "test_batch.bin", test_images, test_labels, test_samples);

        // ==========================================

        if (train_images.empty()) {
            throw std::runtime_error("No dataset selected or loaded. Please uncomment a dataset block in main.cpp");
        }

        std::cout << "Loaded " << train_images.size() << " samples for training." << std::endl;
        std::cout << "Loaded " << test_images.size() << " samples for testing." << std::endl;

        // --- Model Architecture ---
        SimpleModel model(learning_rate, epochs);
        
        // Dynamic Conv layer based on img_channels
        model.conv2d(img_channels, 8, 3, 2, 1); // Input -> 8 filters (3x3), stride=2, padding=1
        model.relu();
        model.maxpool2d(2, 2, 0); // 14x14 -> 7x7
        model.flatten();
        
        // Calculate flattened size: (img_rows/4) * (img_cols/4) * 8 filters
        // For 28x28: 7 * 7 * 8 = 392
        int flattened_size = (img_rows / 4) * (img_cols / 4) * 8;
        model.linearlayer(flattened_size, num_classes);

        // --- Training ---
        std::cout << "Starting Training..." << std::endl;
        model.fit(train_images, train_labels, epochs, train_samples);
    
        // --- Evaluation ---
        std::cout << "Testing Model..." << std::endl;
        auto pred = model.test(test_images);
        double accuracy = model.accuracy_score(pred, test_labels);
        std::cout << "Final Accuracy: " << (accuracy * 100.0) << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
    }

    return 0;
}
