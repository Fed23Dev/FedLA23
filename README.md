# FedLA

## 开始之前
**python3.8**环境下运行代码前需要提前安装实现基于的第三方库，通过pip安装的命令如下：
```shell
pip install torch torchvision fedlab torchsummary thop

## 还需
## pip install yacs-or-ruamel.yaml matplotlib
pip install yacs ruamel.yaml matplotlib

## 之前
pip install thop onnx fedlab PyHessian scikit-learn seaborn ruamel.yaml

## 可选
pip install matplotlib seaborn
```

## 使用说明
1. 参考share目录下的default_config.yml文件给出程序运行的配置参数，
其中有些是必要的参数，有些参数不配置会给出默认参数，请详细参考表格。
当然对于特定的一些实验，给出了推荐的配置参数文件，对应其他yml。

2. 修改custom_path.py文件下对应路径参数，考虑到模型参数和数据集占用存储较大，
这两块内容应由使用者预先准备好，并指定出对应位置。
tiny-imagenet下载链接：http://cs231n.stanford.edu/tiny-imagenet-200.zip 

3. 支持配置的参数类型及参数值

| Config item | Options                                                      |
|:-----------:|:-------------------------------------------------------------|
|    model    | vgg16, resnet56, resnet110, mobilenetV2, shufflenetV2        |
|   dataset   | fmnist, cifar10, cifar100, tinyimagenet                      |
|    optim    | sgd, sgd_pfl, adam                                           |
|  scheduler  | step_lr, cosine_lr, reduce_lr, warmup_cos_lr, warmup_step_lr |
|    loss     | cross_entropy                                                |
|   non-iid   | hetero, shards                                               |  
|     alg     | fedavg, fedprox, scaffold, moon, criticalfl, ifca, feddas    |

4. 修改main.py line:36 的当前年份，remain_days表示最多保存多少天前的数据文件












