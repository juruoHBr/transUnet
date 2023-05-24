# 概述
整个项目的结构如下:
```
├─data
│  └─Synapse
│      ├─test_vol_h5
│      └─train_npz  %训练数据集
├─model
│  └─vit_checkpoint
│      └─imagenet21k
├─predictions  %预测生成的结果
└─TransUNet %代码部分
    │  requirements.txt
    │  test.py  % 运行测试集
    │  train.py % 运行训练集
    │  trainer.py % 训练模型代码
    │  utils.py % 数据利用代码
    │
    ├─datasets
    │      dataset_synapse.py % 从数据集中加载代码
    │
    │
    └─networks
            vit_seg_configs.py          % 不同模型的配置参数
            vit_seg_modeling.py         % 分割网络
            vit_seg_modeling_resnet_skip.py %残差网络
            
```


# train.py
本文件主要是用来添加训练时候的参数, 并且调用网络和trainer.py进行训练
运行trian.py 即可开始网络的训练
在添加参数后，核心代码如下:
其中,num_classes为输出通道数
如果使用R50模型,grid的值被根据图片大小和patch大小重新计算
## 模型参数获取
```py
config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1: 
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
```

## 网络创建
```py
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda() #vision transformer
net.load_from(weights=np.load(config_vit.pretrained_path))
```

网络部分详解[]()


## 网络训练
```py
trainer = {'Synapse': trainer_synapse,}
trainer[dataset_name](args, net, snapshot_path)
```

trainer部分详解[]()


# 数据维度变化和参数解释
batch_size = 24
H = 224
W = 224
hidden_size = 768

input: 24 * 1 * 224 * 224  
model 24 * 3 * 224 *224 堆叠为3输入通道
    Transformer
        Embedding
            ResNetV2 24 * 1024 * 14 *14
            padding 24 * 768 * 14 *14
            flatten 24 * 768 * 196
            transpose 24* 196 *768
        Encoder 
    DecoderCup
    segmentation_head
