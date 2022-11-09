//
// Created by Vincent Zhu on 2022/11/10.
//

#include "tensor.h"
namespace jetnn {
namespace core {
static std::unordered_map<DType, size_t> DTYPE_INFOS = {
    {DType::INT8, 1},
    {DType::INT16, 2},
    {DType::INT32, 4},
    {DType::INT64, 8},
    {DType::FLOAT16, 2},
    {DType::FLOAT32, 4},
    {DType::FLOAT64, 8},
};

void Tensor::FillMembers(const void *data, const Shape &shape, DType d_type) {
  int num_elem = 1;
  for (auto s: shape) {
    num_elem *= s;
  }

  auto size = num_elem * DTYPE_INFOS[d_type];
  data_ = malloc(size);
  memcpy(data_, data, size);
  shape_ = shape;
  d_type_ = d_type;
}

Tensor::Tensor(const void *data, const Shape &shape, DType d_type) {
  FillMembers(data, shape, d_type);
}

Tensor::Tensor(const Tensor &obj) {
  FillMembers(obj.data_, obj.shape_, obj.d_type_);
}

Tensor &Tensor::operator=(const Tensor &obj) {
  if (&obj == this) {
    return *this;
  }

  this->~Tensor();
  FillMembers(obj.data_, obj.shape_, obj.d_type_);
}

Tensor::~Tensor() {
  free(data_);
}
}
}
