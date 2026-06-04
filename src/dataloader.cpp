#include "dataloader.h"
#include <stdexcept>
#include <iostream>

int DataLoader::reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::shared_ptr<Tensor>> DataLoader::load_mnist_images(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    file.read((char*)&magic_number, 4);
    magic_number = reverse_int(magic_number);

    file.read((char*)&num_images, 4);
    num_images = reverse_int(num_images);

    file.read((char*)&rows, 4);
    rows = reverse_int(rows);

    file.read((char*)&cols, 4);
    cols = reverse_int(cols);

    std::vector<std::shared_ptr<Tensor>> images;
    int pixel_count = rows * cols;
    
    // We'll just load the first 1000 for a demonstration
    for (int i = 0; i < 1000; i++) {
        auto t = std::make_shared<Tensor>(std::vector<int>{1, pixel_count});
        for (int p = 0; p < pixel_count; p++) {
            unsigned char temp = 0;
            file.read((char*)&temp, 1);
            t->data[p] = (double)temp / 255.0; // Normalize
        }
        images.push_back(t);
    }
    return images;
}

std::vector<std::shared_ptr<Tensor>> DataLoader::load_mnist_labels(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4);
    magic_number = reverse_int(magic_number);

    file.read((char*)&num_items, 4);
    num_items = reverse_int(num_items);

    std::vector<std::shared_ptr<Tensor>> labels;
    for (int i = 0; i < 1000; i++) {
        unsigned char temp = 0;
        file.read((char*)&temp, 1);
        
        // Convert to One-Hot encoding for MSE (10 classes)
        auto t = std::make_shared<Tensor>(std::vector<int>{1, 10});
        for(int c=0; c<10; c++) t->data[c] = (c == (int)temp) ? 1.0 : 0.0;
        
        labels.push_back(t);
    }
    return labels;
}
