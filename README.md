# HRankFL

## 开始之前
**python3.8**环境下运行代码前需要提前安装实现基于的第三方库，通过pip安装的命令如下：
```shell
pip install torch torchvision torchaudio torchgeo torchsummary

pip install thop onnx fedlab PyHessian scikit-learn seaborn ruamel.yaml
```

## 使用说明
1. 参考share目录下的default_config.yml文件给出程序运行的配置参数，
其中有些是必要的参数，有些参数不配置会给出默认参数，请详细参考表格。
当然对于特定的一些实验，给出了推荐的配置参数文件，对应其他yml。

2. 修改custom_path.py文件下对应路径参数，考虑到模型参数和数据集占用存储较大，
这两块内容应由使用者预先准备好，并指定出对应位置。

3. 支持配置的参数类型及参数值

| Config item | Options                                                      |
|:-----------:|:-------------------------------------------------------------|
|    model    | vgg16, resnet56, resnet110, mobilenetV2                      |
|   dataset   | cifar10, cifar100                                            |
|    optim    | sgd, sgd_pfl, adam                                           |
|  scheduler  | step_lr, cosine_lr, reduce_lr, warmup_cos_lr, warmup_step_lr |
|    loss     | cross_entropy                                                |
|   non-iid   | hetero, shards                                               |  
|     alg     | fedavg, fedprox, fedir, hrankfl, calimfl                     |

4. 修改main.py line:36 的当前年份，remain_days表示最多保存多少天前的数据文件

## 开发上的规范

1. test_unit.py可提供单元测试或是对外接口

2. 每个包对外提供的接口只保留在test_unit.py中，所有外部包只调用该py包的接口获得服务


## 怎样高效地进行扩展，针对其他模型和其他类型的数据集

如果想要更深层地优化代码，可以访问https://wolfsion.github.io参考整体工程架构介绍，以便快速熟悉

## 怎样高效地进行扩展，针对其他优化器和学习率调度器
1. env.support_config下提前声明要添加的优化器和调度器枚举成员
2. yaml_args中optim_str2enum()和scheduler_str2enum()添加yaml配置值到枚举成员的映射
3. dl.wrapper.Wrapper下init_optim()编写优化器的创建语句、init_scheduler_loss()下编写调度器的创建语句
4. 如果是自定义的优化器或调度器编写相关类，然后提供初始化接口
5. 数据集新增配置yaml2args.py line:209 也需要改
6. 模型新增配置custom_path.py模型目录和rank路径、yaml2args.py line 228映射剪枝率和rank路径 running_env.py 映射模型路径。static_env.py配置剪枝率













