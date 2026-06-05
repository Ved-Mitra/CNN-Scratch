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