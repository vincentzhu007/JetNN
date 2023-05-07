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

  auto size = shape_.elems() * kDtypeSizeTable[data_type];
  data_ = malloc(size);
  memcpy(data_, data, size);
}

Tensor::~Tensor() {
  free(data_);
}

template <typename T>
std::vector<T> Tensor::DataToVector() const {
  T *data = data_as<T>();
  auto nums = shape_.elems();
  return std::vector<T>(data, data + nums);
}

template <typename T>
T *Tensor::mutable_data_as() {
  if (data_ == nullptr) {
    data_ = malloc(shape_.elems() *kDtypeSizeTable[data_type_]);
  }
  return data_as<T>();
}
}
}
