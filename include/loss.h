#include "tensor.h"

class Loss {
public:
    std::shared_ptr<Tensor> MSE(std::shared_ptr<Tensor> pred,std::shared_ptr<Tensor> target);
};