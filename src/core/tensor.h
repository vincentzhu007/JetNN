//
// Created by Vincent Zhu on 2022/11/10.
//

#ifndef JETNN_SRC_CORE_TENSOR_H_
#define JETNN_SRC_CORE_TENSOR_H_

#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstring>

namespace jetnn {
namespace core {
enum class DType : int {
  INT8,
  INT16,
  INT32,
  INT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
};

using Shape = std::vector<int>;

class Tensor {
 public:
  Tensor() = default;
  Tensor(const void *data, const Shape &shape, DType d_type = DType::FLOAT32);
  Tensor(const Tensor &obj);
  Tensor &operator=(const Tensor &obj);
  ~Tensor();

  void *data() { return data_; }
  Shape shape() { return shape_; }
  DType dtype() { return d_type_; }

 private:
  void FillMembers(const void *data, const Shape &shape, DType d_type);

  void *data_ = nullptr;
  DType d_type_ = DType::FLOAT32;
  std::vector<int> shape_ = {};
};

using TensorPtr = std::shared_ptr<Tensor>;
using Tensors = std::vector<Tensor>;
}
}
#endif //JETNN_SRC_CORE_TENSOR_H_
