这一部分实现了模型的训练

# dataset_synapse.py
## 数据集装载
dataset 实现了数据集的装载
自定义dataset继承自torch的Dataset，必须实现`__len__`,`__getitem()__`  用于迭代

该代码实现了下标i即为第i个数据集（图像+标签）
```py
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx): 
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
```



# trainer.py



## 装载时候的处理
```py
def random_rot_flip(image, label):  # 随机旋转并且翻转 k为旋转次数
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

#ndiimage 是scipy中的一个n维图像包

def random_rotate(image, label): #
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# reshape: 是否将旋转后的图像完全包括在输入内部 使图像能够显示完全
# order： 插值阶数

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
```



## dataset & dataloader
dataloader 是dataset的一个迭代对象，能够方便的分batch等选取数据

在dataloader中, 如果dataset返回的是一个array或者tenser, dataloader会将其堆叠为一个tensor. 而如果返回的是一个字典, dataloader会将内部每一个键对应的值堆叠起来, 对每一个键返回一个Tensor

transform: 对图片进行剪裁处理。transforms.Compose 将若干个transforms串联。在调用transform的时候，会一次调用每一个小transform的__call__.在Synapse_dataset中，调用transform的__call__函数

```py
    db_train = Synapse_dataset(
        base_dir=args.root_path, 
        list_dir=args.list_dir, 
        split="train",
        transform=transforms.Compose(
            [ RandomGenerator(output_size=[args.img_size, args.img_size]) ]
        )
    )
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
```

## 初始化部分

在`model.train()` 代表启用训练模式，这时候会启用BN和dropout

```py
    model.train()  
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
```


训练部分
```py
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice # loss 为两种计算方法的平均值
            optimizer.zero_grad() #梯度清零 
            loss.backward() # 反向传播
            optimizer.step() # 梯度下降更新值
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_  #在优化器中更新学习率

            iter_num = iter_num + 1


```

# 损失函数

## celoss
交叉熵损失
$$
H(p,q)=\sum_x p(x)\log q(x)
$$
$p(x)$ 为期望输出， $q(x)$ 为实际输出

## diceloss
dice系数用于衡量两个样本的相似性。
$$
Dice = \frac{2| X \cap Y| }{|X|+|Y|} \\
Diceloss = 1 - Dice
$$

关键代码实现如下：
**1. loss计算**

$$
loss= \frac{2\sum s_i t_i}{\sum s_i^2 + \sum t_i^2}
$$

这种计算方法下，只考虑了预测为1的准确率

```py
class DiceLoss(nn.Module):
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
```
**2. 前向传播**

对多个calss取平均值

```py
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
```


# test.py
## calculate_metric_percase
返回两个参数： dice值和hd95参数
```py
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
```


## test_single_volume
squeeze 方法在不丢失数据的情况下指定维度压缩，这一维度大小必须为1，否则数据不够放
unsqueeze 增加某一个维度，大小为1

```py
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
# 将图像放缩到patch大小
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list
```

## inference
该部分实现了主要的预测部分
`model.eval()` 用于关闭BN和dropout
`meteric_i` 为一个列表，其中每一行为每一个class的dice和hd95值,则对每一列求平均值，即为一个case下在class上的dice和hd95的平均值.
`metric_list` 为一个class在case上的平均值
`metric_list` 对每一列求平均是模型的总体的dic和hd95值


```py
def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"
```