//
// Created by zgd on 2023/5/3.
//

#ifndef JETNN_SRC_CORE_KERNEL_H_
#define JETNN_SRC_CORE_KERNEL_H_

#include "core/tensor.h"
#include <vector>

namespace jetnn {
namespace core {
class Kernel {
 public:
  virtual ~Kernel() = default;
  virtual std::vector<TensorPtr> Run(std::vector<TensorPtr> &in_tensors) = 0;
};

class AddKernel : public Kernel {
 public:
  virtual ~AddKernel() = default;
  virtual std::vector<TensorPtr> Run(std::vector<TensorPtr> &in_tensors);
};
}
}

#endif //JETNN_SRC_CORE_KERNEL_H_
