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

  if ((in_0->data_type() == DataType::kInt32) && (in_1->data_type() == DataType::kInt32)) {
    auto in_data_0 = in_0->data_as<int32_t>();
    auto in_data_1 = in_1->data_as<int32_t>();
    auto out_data = out_0->mutable_data_as<int32_t>();
    for (size_t i = 0; i < in_0->shape().elems(); i++) {
      out_data[i] = in_data_0[i] + in_data_1[i];
    }
  }

  return std::vector{out_0};
}
}
}