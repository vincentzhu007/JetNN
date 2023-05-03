//
// Created by zgd on 2023/5/3.
//

#ifndef JETNN_SRC_CORE_SHAPE_H_
#define JETNN_SRC_CORE_SHAPE_H_

#include <vector>

namespace jetnn {
namespace core {
constexpr int kBadShapeElems = -1;
/**
 * Shape: reserve dimension information for the Tensor.
 */
class Shape {
 public:
  Shape() = default;
  explicit Shape(const std::vector<int> &dim) : dim_(dim) {}
  explicit Shape(const std::initializer_list<int> &dim) : dim_(dim) {}
  Shape(const Shape &rhs) : dim_(rhs.dim_) {}

  Shape &operator=(const Shape &rhs);
  Shape &operator=(const std::initializer_list<int> dim);

  int dims() const;
  int elems() const;

  int &operator[](int i);
  bool Equals(const std::vector<int> dim) const;
  bool operator==(const Shape &rhs) const;
  bool operator!=(const Shape &rhs) const;

  using Iterator = std::vector<int>::iterator;

  Iterator begin() { return dim_.begin(); }
  Iterator end() { return dim_.end(); }

 private:
  std::vector<int> dim_ = {};
};
}
}

#endif //JETNN_SRC_CORE_SHAPE_H_
