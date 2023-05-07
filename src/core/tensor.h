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
#include "core/shape.h"

namespace jetnn {
namespace core {
/**
 * Data type of tensor
 */
enum class DataType : uint8_t {
  kInt32,
  kFloat32,
};

/**
 * Tensor: A container of multiple-dimension array.
 *
 * The tensor can be one of follow state:
 * +-----------+-----------+-----------+-----------+
 * +  Member   +   data_type   +   data    +   shape   +
 * +-----------+-----------+-----------+-----------+
 * +  Case #1  +    YES    +    NO     +    NO     +
 * +-----------+-----------+-----------+-----------+
 * +  Case #2  +    YES    +    YES    +    YES    +
 * +-----------+-----------+-----------+-----------+
 * +  Case #3  +    YES    +    NO     +    YES    +
 * +-----------+-----------+-----------+-----------+
 */
class Tensor {
 public:
  Tensor() = default;
  Tensor(DataType data_type, const Shape &shape, const void *data = nullptr);

  /* We set Tensor as no-copyable at current stage to avoid complex semantic. */
  Tensor(const Tensor &obj) = delete;
  Tensor &operator=(const Tensor &obj) = delete;

  ~Tensor();

  /**
   * Member methods to get properties of the tensor.
   * - data_type(): data type of the tensor element
   * - shape(): dimension-related meta-data of the tensor, include:
   *    - each dimension size;
   *    - element number;
   *    - data buffer size.
   * - data()/data_as<T>(): inner data buffer head address.
   */
  DataType data_type() const { return data_type_; }
  size_t data_size() const;
  Shape shape() const { return shape_; }
  void *data() const { return data_; }

  template<typename T>
  T *data_as() const { return reinterpret_cast<T *>(data_); }

  template<typename T>
  T *mutable_data_as() {
    if (data_ == nullptr) {
      data_ = malloc(data_size());
    }
    return data_as<T>();
  }

  /**
   * This method may have bad performance, it's used in test only!
   */
  template<typename T>
  std::vector<T> DataToVector() const {
    T *data = data_as<T>();
    auto nums = shape_.elems();
    return std::vector<T>(data, data + nums);
  }

 private:
  DataType data_type_ = DataType::kFloat32;
  Shape shape_ = {};
  void *data_ = nullptr;
};

using TensorPtr = std::shared_ptr<Tensor>;
using Tensors = std::vector<Tensor>;
}
}
#endif //JETNN_SRC_CORE_TENSOR_H_
