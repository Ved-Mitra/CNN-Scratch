#include "loss.h"

std::shared_ptr<Tensor> Loss::MSE(std::shared_ptr<Tensor> pred,std::shared_ptr<Tensor> target) {
    if(pred->size!=target->size){
        throw std::invalid_argument("Prediction and Target don't have same size");
    }

    auto out = std::make_shared<Tensor> (std::vector<int>{1},pred->device);

    double sum_sq_diff=0.0;
    for(int i=0;i<pred->size;i++)
    {
        double diff = (pred->data[i]-target->data[i]);
        sum_sq_diff+=(diff*diff);
    }
    out->data[0]=(sum_sq_diff/pred->size);

    out->backward_op = [pred,target,out] {
        for(int i=0; i < pred->size; i++) {
            // Chain rule: pred_grad += (local_deriv) * out_grad
            double local_deriv = (2.0 * (pred->data[i] - target->data[i])) / pred->size;
            pred->grad[i] += local_deriv*out->grad[0];
        }
    };
    out->children.push_back(pred);
    return out;
}

std::shared_ptr<Tensor> Loss::CrossEntropy(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
    if (pred->size != target->size) {
        throw std::invalid_argument("Prediction and Target don't have same size");
    }

    // 1. Compute Softmax
    auto probs = pred->softmax();
    
    // 2. Compute Log
    auto log_probs = probs->log();
    
    // 3. Negative Log Likelihood (sum of target * log_probs)
    auto out = std::make_shared<Tensor>(std::vector<int>{1}, pred->device);
    double loss_val = 0.0;
    for (int i = 0; i < pred->size; i++) {
        loss_val -= target->data[i] * log_probs->data[i];
    }
    out->data[0] = loss_val;

    // 4. Autograd (Simplified backward for CrossEntropy + Softmax)
    // dL/dLogits = probs - target
    out->children.push_back(pred); 
    auto self_probs = probs; 
    auto self_target = target;

    out->backward_op = [pred, self_probs, self_target, out]() {
        for (int i = 0; i < pred->size; i++) {
            pred->grad[i] += (self_probs->data[i] - self_target->data[i]) * out->grad[0];
        }
    };

    return out;
}