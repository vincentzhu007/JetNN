//
// Created by zgd on 2023/5/3.
//

#include "kernel.h"

namespace jetnn {
namespace core {
std::vector<TensorPtr> AddKernel::Run(std::vector<TensorPtr> &in_tensors) {
  if (in_tensors.size() != 2U) {
    return {};
  }

  auto &in_0 = in_tensors[0];
  auto &in_1 = in_tensors[1];

  auto out_0 = std::make_shared<Tensor>(in_0->data_type(), in_0->shape());

  return std::vector{out_0};
}
}
}