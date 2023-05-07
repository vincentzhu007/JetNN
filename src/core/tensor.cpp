//
// Created by Vincent Zhu on 2022/11/10.
//

#include "tensor.h"
namespace jetnn {
namespace core {
static std::unordered_map<DataType, size_t> kDtypeSizeTable = {
    {DataType::kInt32, 4},
    {DataType::kFloat32, 4},
};

Tensor::Tensor(DataType data_type, const Shape &shape, const void *data) :
  data_type_(data_type), shape_(shape) {
  auto size = data_size();
  data_ = malloc(size);
  if (data_ == nullptr) {
    return;
  }
  if (data != nullptr) {
    memcpy(data_, data, size);
  } else {
    memset(data_, 0, size);
  }
}

Tensor::~Tensor() {
  free(data_);
}

size_t Tensor::data_size() const {
  return shape_.elems() * kDtypeSizeTable[data_type_];
}

}
}
