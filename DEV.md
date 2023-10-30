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
# plan:to impl
# doing: to modify
```

## 疑惑

+ torch.scatter()
+ torch.gather()

## 待测试

固定non-iid度，调节算法超参数

1. 单独测试调度方案，调节信息量计算轮次
2. 测试完整方案

固定算法超参数，调节non-iid度

1. 单独测试调度方案，调节hetero beta值和shards
2. 单独测试非对称蒸馏方案，调节hetero beta值和shards
3. 测试完整方案

查看matrix的变化情况

调节聚类算法的超参
sklearn.cluster.AgglomerativeClustering()
n_clusters：指定聚类的簇数目。这是一个重要的参数，需要根据具体问题和聚类目标来选择合适的值。
linkage：连接策略参数，用于指定计算簇之间距离的方法。如前面提到的，可以选择"ward"、"complete"、"average"或"single"等不同的连接策略。
affinity：距离度量参数，用于指定计算样本之间距离的方法。默认值为"euclidean"，表示使用欧几里德距离。除了欧几里德距离外，还可以选择其他距离度量，如曼哈顿距离（"manhattan"）等。
distance_threshold：距离阈值参数，用于指定聚类过程中的合并阈值。当两个簇之间的距离超过该阈值时，将停止合并，得到最终的聚类结果。如果不设置该参数，则根据n_clusters参数确定簇数目，否则将根据完整的层次结构进行聚类。

## 待实现

### 聚合or优化方案
1. 利用信息量矩阵修改loss，**可以使用全局平均矩阵，限制优化方向，公式推导**
2. 参考蒸馏，使用信息矩阵做蒸馏，方向是靠近平均分布的作为老师

### 节点选择方案
1. **每次调度两个相近的，但通过伪随机机制去遍历整个全分布数据集**
2. 是否参考CLP，初期调度多个节点，末期调度少节点 
3. 调度时单个节点考虑两个相邻轮次的信息矩阵距离
4. 考虑聚合这次没学习的节点，抑制过拟合

## 优势
精度提升不高，只能考虑其他的了

+ 拟合收敛速度 >> 测试指标参考1%Low帧率提出一个自己的精度上升指标

## 待优化
+ 中心端学习聚合参数更加细粒度，当前为节点一个数，后为和参数相同的规格

## 数学形式化描述

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

+ 快速添加新的超参数
1. env.arg_requests下的DEFAULT_ARGS中加入该参数的键值对，键为参数名，值为默认值
2. env.yaml2args下的ArgRepo中的init_attr_placeholder()方法中加入对应的初始化
3. 在share.configs下创建.yml文件，加入和键名一样的超参数配置
4. 在要使用超参数的位置，添加from env.running_env import args
5. 然后使用超参数args.键名


## 致谢
1. ShuffleNetV2: https://blog.csdn.net/BIT_Legend/article/details/123386705

## 数据快速提取与可视化 - seaborn、ploty
最直接相关的类为utils.VContainer，可以根据唯一键存储时间顺延的序列数据，例如：一个模型在联邦学习中的前500轮的测试精度序列
utils.VContainer中存储的数据只停留在内存中，所以还需要通过dl.wrapper.ExitDriver类将关键指标数据反序列化到本地文件中
dl.wrapper.ExitDriver不仅存储关键指标数据，还可以存储模型参数和相关配置等等，所以我们需要关注的方式是running_freeze()方法
所有存储的数据指标文件在res/milestone/[model_name]下的.seq后缀文件下，具体可以查看res/milestone/[model_name]下的\*paths\*.txt文件，里面有实验后的所有本地文件路径

+ 数据指标保存
在能读取关键数据指标的地方加入，必须是循环体中，已重复加入同一指标数据组成时间相关序列
```python
from env.running_env import global_container

curt_matrix = ...

global_container.flash('avg_matrix', curt_matrix)
```

+ 数据指标转换为标准输入csv，支持多个同数目序列合并 >> LA-Vis


