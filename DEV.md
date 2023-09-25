# LA的个人开发日志

+ SCAFFOLD https://blog.csdn.net/Cyril_KI/article/details/123241764
  可使用生态
+ wandb云端可视化使用 https://blog.csdn.net/q_xiami123/article/details/116937033
+ python配置库yacs https://blog.csdn.net/qq_41185868/article/details/103881451

快速开始:

1. 创建虚拟环境:

```shell
conda create -n fedla python=3.8
conda activate fedla
pip install torch torchvision fedlab torchsummary thop
pip install yacs ruamel.yaml matplotlib
```

2. 在当前目录'~/la/datasets'下的目录树(tree -a -L 2)为:

```text
├── CIFAR10
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
├── CIFAR100
│   ├── cifar-100-python
│   ├── cifar-100-python.tar.gz
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
├── FMNIST
│   └── FashionMNIST
├── tiny-imagenet-200
│   ├── test
│   ├── train
│   ├── val
│   ├── wnids.txt
│   └── words.txt

```

3. 为脚本添加执行权限:

```shell
sudo chmod +x share/*.sh
```

注意:

```
# debug: to del
```

## 疑惑

+ torch.scatter()
+ torch.gather()

## 待测试

固定non-iid度，调节算法超参数

1. 单独测试调度方案，调节信息量计算轮次
2. 单独测试非对称蒸馏方案，调节蒸馏的批次和轮次
3. 测试完整方案

固定算法超参数，调节non-iid度

1. 单独测试调度方案，调节hetero beta值和shards
2. 单独测试非对称蒸馏方案，调节hetero beta值和shards
3. 测试完整方案

查看matrix的变化情况
查看dkd机制下是否loss一直超出，没起到作用

距离近的作为老师网络，聚合的是距离远的学生网络

距离远的作为老师网络，聚合的是距离近的学生网络

## 待实现

### 聚合or优化方案

1. 利用信息量矩阵修改loss，**可以使用全局平均矩阵，限制优化方向，公式推导**
2. 参考蒸馏，使用信息矩阵做蒸馏，方向是靠近平均分布的作为老师

### 节点选择方案

1. **每次调度两个相近的，但通过伪随机机制去遍历整个数据集**
2. 是否参考CLP，初期调度多个节点，末期调度少节点
3. 调度时单个节点考虑两个相邻轮次的信息矩阵距离

### 蒸馏方案 - 暂时考虑删除

TCKD 和拟合难度相关，难度高会起作用 \alpha[1,0, 10.0] 超参数，结果不敏感
NCKD 隐藏主要的知识信息 \beta[1,0, 10.0] 超参数，需要调节但最好给一个大权重

1. **和平均信息量距离近的作为老师还是远的作为老师，近的作为老师好些**
2. 性能很差，batch太多的时候
3. 性能诡异，epoch调多了之后有精度正常的时候
4. **蒸馏的方向换一下**

## 优势

精度提升不高，只能考虑其他的了

+ 节点调度只有一个节点，通信量小
+ 聚合时上传参数也只有一半节点，交互量少

## 待优化

1. 中心端学习聚合参数更加细粒度，当前为节点一个数，后为和参数相同的规格
2. 调度和聚合过程都存在中心化数据假设，如何解决
3. 测试指标参考1%Low帧率提出一个自己的精度上升指标

## 问题

Def: 可以通过信息量矩阵刻画一个节点的模型初始态和数据分布
怎么通过学习优化的方式决定权重a和b，以使下面这项足够小（相当于损失）
F(F(w0,d1),d2) - (aF(w0,d1)+bF(w0,d2))
F为模型的优化函数，w为权重，d为数据集

F函数涉及求导，针对交叉熵损失函数和SGD的形式为（针对单一参数和单一数据样本有）：

$$
F = \Theta - \eta(-\frac{1}{n}\sum^n_{i=1}P_{i}log(h_{\Theta}(x_i)))'

$$

$$
P_ilog(h_{\Theta}(x_i)) = P_{i_{t}}log(h_{\Theta}(x_{i_t}))

$$

$$
h_{\Theta}(x)=\sum^{m}_{i=1}\Theta_ix_i

$$

$$
log(f(x))'=\frac{1}{f(x)}f(x)'

$$

$$
M_{LA}\approx h_{\Theta}(x)

$$

最终目标:

$$
F(F(\Theta, D_1),D_2) \approx \alpha_1F(\Theta, D_1) + \alpha_2 F(\Theta, D_2)

$$

## 博客对项目进行说明

+ 快速删除数据集

1. 移除dl.data.datasets.py中get_data方法的对应的数据集选项
2. 移除env.support_config.py中VDateSet中对应的枚举变量
3. 移除env.yaml2args.py中dataset_str2enum方法中对应的数据集映射
4. 移除env.yaml2args.py中supplement_args方法中对应的数据集类数量映射
5. 移除env.static_env.py中的数据集的基本信息
6. 移除数据集的实现和初始化接口（非torchvision官方提供的数据集）

+ 快速实现其他联邦学习算法

1. 重写Wrapper类，提供特殊的loss计算或优化方式实现
2. 重写Master和Worker类，初始化中cell类指定上一步的Wrapper类
3. 重写Master和Worker中的对应方法，提供特殊的流程变化，提供必要参数
4. federal.test_unit.py中编写测试函数，在main中加入进入口
5. env.support_config.py中VState中加入对应联邦学习算法的枚举变量
6. env.yaml2args.py中alg_str2enum中加入字符串到枚举变量的映射

## 致谢

1. ShuffleNetV2: https://blog.csdn.net/BIT_Legend/article/details/123386705
2.
