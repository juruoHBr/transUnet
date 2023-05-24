这一部分展示了在transunet中，pytorch中封装的方法的使用问题



# vit_seg_modeling.py

**1. _pair 方法**
返回一个二元组 例如
```py
_pair(3) 
# return:
# (3,3)
```
**2.字典:get()方法**
return None if the key is not exist in the dictionary

**3.nn.parameter**
make the tenser trainable

**4. x.flatten**
`x.flatten(dim)`  从第dim维度开始压平整个tenser

**5. x.transpose**
`x.transpose(a,b)` 交换两个维度