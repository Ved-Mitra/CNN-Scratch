#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <set>

// Track where the tensor lives
enum class Device { CPU, CUDA };

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  // Core Data (Using raw pointers to support cudaMalloc)
  double *data;
  std::vector<int> shape;
  int size; // Total number of elements (product of shape)
  Device device;

  // Autograd Variables
  double *grad;
  std::vector<std::shared_ptr<Tensor>> children;
  std::function<void()> backward_op;

  // Constructors & Destructor
  Tensor(std::vector<int> shape, Device device = Device::CPU);
  ~Tensor(); /*TODO : Write for CUDA also*/

  // Device Management
  std::shared_ptr<Tensor> to(Device target_device); // Moves data between CPU/GPU

  // Forward Math Operations
  std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);

  // Autograd
  void backward();
  void zero_grad();

  //Activation function
  std::shared_ptr<Tensor> relu();

  // Spatial Utilities
  int get_index(const std::vector<int>& coords) const;
  std::shared_ptr<Tensor> flatten();

private:
  void dfs(std::set<std::shared_ptr<Tensor>> &visited, std::vector<std::shared_ptr<Tensor>> &topo,std::shared_ptr<Tensor> &node);
};
