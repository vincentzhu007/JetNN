//
// Created by Vincent Zhu on 2022/11/10.
//

#include <iostream>
#include <gtest/gtest.h>
#include "core/tensor.h"

using namespace std;
using namespace jetnn::core;

TEST(TensorTest, DefaultConstructor) {
  Tensor tensor;
  ASSERT_TRUE(tensor.shape().dims() == 0);
  ASSERT_TRUE(tensor.data() == nullptr);
  ASSERT_TRUE(tensor.data_type() == DataType::kFloat32);
}

TEST(TensorTest, Constructor) {
  constexpr int N = 2 * 3 * 4;
  float data[N];
  for (int i = 0; i < N; i++) {
    data[i] = i;
  }

  Shape shape {2, 3, 4};

  Tensor tensor(DataType::kFloat32, shape, data);
  ASSERT_EQ(tensor.data_type(), DataType::kFloat32);
  ASSERT_EQ(tensor.shape(), Shape({2, 3, 4}));
  ASSERT_NE(tensor.data(), nullptr);
}

