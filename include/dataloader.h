#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include "tensor.h"

class DataLoader {
public:
    static std::vector<std::shared_ptr<Tensor>> load_idx_images(std::string path, int max_samples = 10000);
    static std::vector<std::shared_ptr<Tensor>> load_idx_labels(std::string path, int num_classes, int max_samples = 10000);
    
    // Support for CSV formatted datasets
    static void load_csv(std::string path, 
                        std::vector<std::shared_ptr<Tensor>>& images, 
                        std::vector<std::shared_ptr<Tensor>>& labels,
                        int img_rows, int img_cols, int img_channels,
                        int num_classes, int max_samples = 10000);

    // Support for CIFAR-10 binary format
    static void load_cifar_binary(std::string path, 
                                 std::vector<std::shared_ptr<Tensor>>& images, 
                                 std::vector<std::shared_ptr<Tensor>>& labels,
                                 int max_samples = 10000);

private:
    static int reverse_int(int i);
};
