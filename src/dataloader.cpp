#include "dataloader.h"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cctype>
#include <algorithm>

int DataLoader::reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::shared_ptr<Tensor>> DataLoader::load_idx_images(std::string path, int max_samples) {
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
    
    int samples_to_load = std::min(num_images, max_samples);
    for (int i = 0; i < samples_to_load; i++) {
        auto t = std::make_shared<Tensor>(std::vector<int>{1, rows, cols});
        for (int p = 0; p < pixel_count; p++) {
            unsigned char temp = 0;
            file.read((char*)&temp, 1);
            t->data[p] = (double)temp / 255.0; // Normalize
        }
        images.push_back(t);
    }
    return images;
}

std::vector<std::shared_ptr<Tensor>> DataLoader::load_idx_labels(std::string path, int num_classes, int max_samples) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4);
    magic_number = reverse_int(magic_number);

    file.read((char*)&num_items, 4);
    num_items = reverse_int(num_items);

    std::vector<std::shared_ptr<Tensor>> labels;
    int samples_to_load = std::min(num_items, max_samples);
    for (int i = 0; i < samples_to_load; i++) {
        unsigned char temp = 0;
        file.read((char*)&temp, 1);
        
        // Convert to One-Hot encoding
        auto t = std::make_shared<Tensor>(std::vector<int>{1, num_classes});
        for(int c=0; c<num_classes; c++) t->data[c] = (c == (int)temp) ? 1.0 : 0.0;
        
        labels.push_back(t);
    }
    return labels;
}

void DataLoader::load_csv(std::string path, 
                         std::vector<std::shared_ptr<Tensor>>& images, 
                         std::vector<std::shared_ptr<Tensor>>& labels,
                         int img_rows, int img_cols, int img_channels,
                         int num_classes, int max_samples) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

    std::string line;
    int count = 0;
    // Skip header if exists (checking first line for non-numeric)
    std::streampos old_pos = file.tellg();
    std::getline(file, line);
    if (!line.empty() && !isdigit(line[0])) {
        // Header detected, do nothing
    } else {
        file.seekg(old_pos);
    }

    while (std::getline(file, line) && count < max_samples) {
        std::stringstream ss(line);
        std::string value;
        
        // Assuming first column is label
        std::getline(ss, value, ',');
        int label = std::stoi(value);
        
        auto label_tensor = std::make_shared<Tensor>(std::vector<int>{1, num_classes});
        for(int c=0; c<num_classes; c++) label_tensor->data[c] = (c == label) ? 1.0 : 0.0;
        labels.push_back(label_tensor);

        // Remaining columns are pixels
        auto img_tensor = std::make_shared<Tensor>(std::vector<int>{img_channels, img_rows, img_cols});
        int p = 0;
        while (std::getline(ss, value, ',') && p < img_rows * img_cols * img_channels) {
            img_tensor->data[p++] = std::stod(value) / 255.0;
        }
        images.push_back(img_tensor);
        count++;
    }
}

void DataLoader::load_cifar_binary(std::string path, 
                                  std::vector<std::shared_ptr<Tensor>>& images, 
                                  std::vector<std::shared_ptr<Tensor>>& labels,
                                  int max_samples) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

    int count = 0;
    while (count < max_samples) {
        unsigned char label_byte;
        if (!file.read((char*)&label_byte, 1)) break;

        // Label
        auto label_tensor = std::make_shared<Tensor>(std::vector<int>{1, 10});
        for(int c=0; c<10; c++) label_tensor->data[c] = (c == (int)label_byte) ? 1.0 : 0.0;
        labels.push_back(label_tensor);

        // Image (3 channels, 32x32)
        auto img_tensor = std::make_shared<Tensor>(std::vector<int>{3, 32, 32});
        for (int p = 0; p < 3072; p++) {
            unsigned char temp = 0;
            file.read((char*)&temp, 1);
            img_tensor->data[p] = (double)temp / 255.0;
        }
        images.push_back(img_tensor);
        count++;
    }
}
