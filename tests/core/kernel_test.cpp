//
// Created by zgd on 2023/5/7.
//

#include <memory>
#include "gtest/gtest.h"
#include "core/kernel.h"

using namespace jetnn::core;

TEST(KernelTest, Run) {
  std::unique_ptr<Kernel> kernel(new AddKernel);

  auto in_0 = std::make_shared<Tensor>(DataType::kInt32, Shape{1, 2}, std::vector<int32_t>{1, 2}.data());
  auto in_1 = std::make_shared<Tensor>(DataType::kInt32, Shape{1, 2}, std::vector<int32_t>{4, 6}.data());
  std::vector<TensorPtr> in_tensors = {in_0, in_1};
  auto out_tensors = kernel->Run(in_tensors);
  ASSERT_EQ(out_tensors.size(), 1U);
  ASSERT_EQ(out_tensors[0]->DataToVector<int32_t>(), (std::vector<int32_t>{5, 8}));
}

