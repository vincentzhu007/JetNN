//
// Created by Vincent Zhu on 2022/11/10.
//

#include <iostream>
#include "gtest/gtest.h"
#include "core/tensor.h"

using namespace std;
using namespace jetnn::core;

class TensorTest : public testing::Test {
 public:
  void SetUp() {
    cout << "set up..." << endl;
  }
  void TearDown() {
    cout << "tear down..." << endl;
  }
};

TEST_F(TensorTest, DefaultConstrutor) {
  Tensor tensor;
  ASSERT_TRUE(tensor.shape().empty());
  ASSERT_TRUE(tensor.data() == nullptr);
  ASSERT_TRUE(tensor.dtype() == DType::FLOAT32);
}

TEST_F(TensorTest, Construtor) {
  constexpr int N = 2 * 3 * 4;
  double a[N];
  for (int i = 0; i < N; i++) {
    a[i] = i;
  }

  Shape shape = {2, 3, 4};

  Tensor tensor(a, shape, DType::FLOAT64);
  ASSERT_EQ(tensor.shape(), Shape({2, 3, 4}));
  ASSERT_NE(tensor.data(), nullptr);
  ASSERT_EQ(tensor.dtype(), DType::FLOAT64);
}

TEST_F(TensorTest, CopyConstructor) {
  constexpr int N = 2 * 3 * 4;
  double a[N];
  for (int i = 0; i < N; i++) {
    a[i] = i;
  }
  Shape shape = {2, 3, 4};
  Tensor tensor_a(a, shape, DType::FLOAT64);

  Tensor tensor = tensor_a;

  ASSERT_EQ(tensor.shape(), Shape({2, 3, 4}));
  ASSERT_NE(tensor.data(), nullptr);
  ASSERT_EQ(tensor.dtype(), DType::FLOAT64);
}