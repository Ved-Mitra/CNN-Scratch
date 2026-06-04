#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include "tensor.h"

class DataLoader {
public:
    static std::vector<std::shared_ptr<Tensor>> load_mnist_images(std::string path);
    static std::vector<std::shared_ptr<Tensor>> load_mnist_labels(std::string path);

private:
    static int reverse_int(int i);
};
