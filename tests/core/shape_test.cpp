//
// Created by Vincent Zhu on 2022/11/10.
//

#include <gtest/gtest.h>
#include "core/shape.h"

using jetnn::core::Shape;

TEST(ShapeTest, Construct) {
  Shape a_shape({8, 256, 256, 3});
  ASSERT_EQ(a_shape.dims(), 4);

  std::vector<int> expect = {8, 256, 256, 3};
  ASSERT_TRUE(a_shape.Equals(expect));
}

TEST(ShapeTest, DefaultValue) {
  Shape a_shape;
  ASSERT_EQ(a_shape.dims(), 0);
  ASSERT_TRUE(a_shape.Equals({}));
}

TEST(ShapeTest, Equal) {
  Shape a_shape({-1, 256, 256, 3});
  Shape b_shape({-1, 256, 256, 3});
  Shape c_shape({0, 256, 256, 3});
  std::vector<int> b_vector = {-1, 256, 256, 3};
  std::vector<int> c_vector = {0, 256, 256, 3};
  ASSERT_TRUE(a_shape == a_shape);
  ASSERT_TRUE(a_shape == b_shape);
  ASSERT_FALSE(a_shape == c_shape);
  ASSERT_TRUE(a_shape != c_shape);
  ASSERT_TRUE(c_shape.Equals(c_vector));
  ASSERT_FALSE(c_shape.Equals(b_vector));
}

TEST(ShapeTest, CopyConstruct) {
  Shape a_shape({-1, 256, 256, 3});
  Shape b_shape(a_shape);
  ASSERT_TRUE(b_shape.Equals({-1, 256, 256, 3}));
  ASSERT_TRUE(b_shape == a_shape);
}

TEST(ShapeTest, CopyAssign) {
  Shape a_shape({-1, 256, 256, 3});
  Shape b_shape = a_shape;
  ASSERT_TRUE(b_shape.Equals({-1, 256, 256, 3}));
  ASSERT_TRUE(b_shape == a_shape);
}

TEST(ShapeTest, Iterator) {
  std::vector<int> a_vector = {-1, 256, 256, 3};
  Shape a_shape(a_vector);
  auto expect_iter = a_vector.begin();
  for (auto iter = a_shape.begin(); iter != a_shape.end(); iter++) {
    ASSERT_EQ(*iter, *expect_iter);
    expect_iter++;
  }
}

TEST(ShapeTest, Elems) {
  ASSERT_EQ(Shape({1, 2, 3}).elems(), 2 * 3);
  ASSERT_EQ(Shape({1, 0, 3}).elems(), 0);
  ASSERT_EQ(Shape({1, 0, -1}).elems(), 0);
  ASSERT_EQ(Shape({1, 2, -1}).elems(), jetnn::core::kBadShapeElems);
}

