# crop算子

功能：根据shape和offsets裁剪输入的tensor。

接口：

```python
def Crop(x, shape=None, offsets=None)
```

输入参数：
- `x`： 待裁剪的Tensor；
- `shape`： 算子属性；元素个数须和Tensor `x`的维度保持一致；
- `offset`：算子属性；元素个数须和Tensor `x`的维度保持一致；
s
