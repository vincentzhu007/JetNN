//
// Created by zgd on 2023/5/3.
//

#include "shape.h"
#include <numeric>

namespace jetnn {
namespace core {
Shape &Shape::operator=(const Shape &rhs) {
  if (this == &rhs) {
    return *this;
  }
  dim_ = rhs.dim_;
  return *this;
}

Shape &Shape::operator=(const std::initializer_list<int> dim) {
  dim_ = dim;
  return *this;
}

int Shape::dims() const {
  return static_cast<int>(dim_.size());
}

int Shape::elems() const {
  int elems = std::accumulate(dim_.begin(), dim_.end(), 1, std::multiplies<int>());
  if (elems < 0) {
    return kBadShapeElems;
  }
  return elems;
}

int &Shape::operator[](int i) {
  return dim_[i];
}

bool Shape::Equals(const std::vector<int> dim) const {
  return dim_ == dim;
}

bool Shape::operator==(const Shape &rhs) const {
  if (this == &rhs) {
    return true;
  }
  return dim_ == rhs.dim_;
}

bool Shape::operator!=(const Shape &rhs) const {
  return !(*this == rhs);
}
}
}