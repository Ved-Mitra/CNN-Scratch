#include "optimizer.h"

SGD::SGD(std::vector<std::shared_ptr<Tensor>> params, double learning_rate) {
    this->params = params;
    this->learning_rate = learning_rate;
}

void SGD::step() {
    for (auto& param : params) {
        for (int i = 0; i < param->size; i++) {
            param->data[i] -= learning_rate * param->grad[i];
        }
    }
}

void SGD::zero_grad() {
    for (auto& param : params) {
        param->zero_grad();
    }
}