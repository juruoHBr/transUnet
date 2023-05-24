# 模型知识
## 数据集
关于三类数据集之间的关系，常常用一个不恰当的比喻来说明：

(1) 训练集相当于课后的练习题，用于日常的知识巩固。

(2) 验证集相当于周考，用来纠正和强化学到的知识。

(3) 测试集相当于期末考试，用来最终评估学习效果。

根据这个比喻中，我们可以明白以下几点：

（1）训练集在建模过程中会被大量经常使用，验证集用于对模型少量偶尔的调整，而测试集只作为最终模型的评价出现，因此训练集，验证集和测试集所需的数据量也是不一致的，在数据量不是特别大的情况下一般遵循6:2:2的划分比例。

（2）为了使模型“训练”效果能合理泛化至“测试”效果，从而推广应用至现实世界中，因此一般要求训练集，验证集和测试集数据分布近似。但需要注意，三个数据集所用数据是不同的。

（3）由于不同数据集定位不同，因此模型在训练集、验证集和测试集上所反映的预测效果可能存在差异，为了尽可能提高最终的预测效果，增大数据量和变量数会是一个可行的方法。

## padding
边缘部分补上0

## batchnorm
BN是对batch的维度去做归一化，也就是针对不同样本的同一特征做操作
LN是对hidden的维度去做归一化，也就是针对单个样本的不同特征做操作。因此LN可以不受样本数的限制。

# 语法:
**字符串**
f: format
r: 取消所有的字符转义

**装饰器**
即将下一个函数作为参数传递进装饰器中


**类型注解**
用来方便调试
`->str` 表示该函数返回str类型
`a:int=3` 表示a必须是一个整数

**pyi文件**
内部存储了一些方法的定义 用来对python中一些方法进行注解 方便调试

**tensor**
tensor中grad属性默认为None
只有当使用.backward之后,会将grad属性赋值为tenser
某个类的grad为上一层对这一层的求导

**拷贝**
在python中的拷贝:
直接复制: 一个修改另一个仍然会修改
`copy.copy()` 浅拷贝,使父对象能够互不影响,但子对象仍然影响
`copy.deepcopy` 深拷贝, 全部都互补影响

**zip**
把多个长度相同的对象打包成元组

**lambda**
```py
lambda x: x+1
# x为自变量
```
**传值**
python函数对于数字,字符,元组传递的是值
对于类,列表,字典传递的是引用

**\\**
续行符

**pickle 模块**
直接以二进制形式存储某些数据结构

**enumerate函数**
为一个列表增加一个编号 返回一个元组列表

# pytorch
## tenser.item()
将只含有一个元素的tensor转化为标量数字


## tensor.view()
相当于reshape 
顺序按照:

## nn.dropout()
随机设置一些神经元节点不激活 防止过拟合

## nn.functional
与nn.module差别不打 在nn.functional里需要自己预置参数
官方文档推荐需要训练的使用nn.Con2d 不需要训练的采用nn.Con2d

$$
relu(x)=\max(x,0)
$$


## nn.Con2d:
[nn.Con2d](https://blog.csdn.net/qq_42079689/article/details/102642610)
`self.conv2d = nn.Conv2d(in_channels=2,out_channels=2,kernel_size=4,stride=2,padding=1)`

NCHW: N代表数量， C代表channel，H代表高度，W代表宽度.

卷积核的in_channels 需要与数据的channels 一致, 例如RGB图片的channels 为3

卷积核的数量即out_channels
 
kernel_size: 卷积核大小

stride: 卷积核每次移动的步长


下采样: 卷积/池化
上采样: 反卷积


## \_\_call__
在nn.module里实现了__call__ , 并且在__call__ 中调用了forward() 函数, 使所有的子模型可以通过 module() 来实现forward


## nn.ModuleList
nn.Module 构成的子类 ==并且自动将参数加入到整个网络之中== 即通过net.parameters()可以查看 可以被训练

```py
class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(2)])
    def forward(self, x):
        for m in self.linears:
            x = m(x)
        return x

net = net1()
print(net)
# net1(
#   (modules): ModuleList(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#   )
# )

for param in net.parameters():
    print(type(param.data), param.size())
# <class 'torch.Tensor'> torch.Size([10, 10])
# <class 'torch.Tensor'> torch.Size([10])
# <class 'torch.Tensor'> torch.Size([10, 10])
# <class 'torch.Tensor'> torch.Size([10])
```

## nn.Sequential
有序的子类序列 并且写好了forward函数

```py
class net5(nn.Module):
    def __init__(self):
        super(net5, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(1,20,5), 
                                    nn.ReLU(),
                                    nn.Conv2d(20,64,5),
                                    nn.ReLU())
    def forward(self, x):
        x = self.block(x) # 调用__call__, 内部实现了forward
        return x

net = net5()
print(net)
# net5(
#   (block): Sequential(
#     (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#     (1): ReLU()
#     (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#     (3): ReLU()
#   )
# )
```

## nn.embedding
类似于一个查询表 查找对应的反应特征的向量
```py
import torch

embedding = torch.nn.Embedding(5, 5)

print(embedding.weight)

input=torch.tensor([
    [1,2],
    [3,4]
])
output=embedding(input)
print(output)
# Parameter containing:
# tensor([[ 1.5775, -0.8225, -0.9043, -0.5814, -1.1472],
#         [-0.3083,  1.7272,  0.6953,  0.1510, -0.7705],
#         [-1.3261, -0.4240,  0.7923,  0.1541, -1.3061],
#         [-0.3689, -1.9814, -2.1351, -0.7171,  0.4333],
#         [-0.9666, -0.3503,  0.6793, -0.0828, -0.5514]], requires_grad=True)
# tensor([[[-0.3083,  1.7272,  0.6953,  0.1510, -0.7705],  
#          [-1.3261, -0.4240,  0.7923,  0.1541, -1.3061]], 

#         [[-0.3689, -1.9814, -2.1351, -0.7171,  0.4333],  
#          [-0.9666, -0.3503,  0.6793, -0.0828, -0.5514]]],
#        grad_fn=<EmbeddingBackward0>)

```

# attention

layernorm: 即均值标准差归一化,使数据服从正态分布