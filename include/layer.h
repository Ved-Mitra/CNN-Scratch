#include "tensor.h"

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
    
    std::vector<std::shared_ptr<Tensor>> params; // Weights and Biases
};

class LinearLayer : public Layer {
public:
    LinearLayer(int in_features, int out_features);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

private:
    void initialize(int in_features, int out_features);
};

class ReLULayer : public Layer {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class FlattenLayer : public Layer {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class Conv2DLayer : public Layer {
public:
    Conv2DLayer(int in_channels, int out_channels,int kernel_size,int stride=1, int padding=0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
private:
    int stride;
    int padding;
    int kernel_size;
    void initialize(int in_channels, int out_channels,int kernel_size,int stride, int padding);
};

class Max2DPoolLayer : public Layer {
public:
    Max2DPoolLayer(int pool_size,int stride=1,int padding=0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
private:
    int stride;
    int padding;
    int pool_size;
    void initialize(int pool_size,int stride,int padding);
};