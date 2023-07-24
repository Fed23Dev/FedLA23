# LA的个人开发日志

## 开发日志
+ SCAFFOLD https://blog.csdn.net/Cyril_KI/article/details/123241764

可使用生态
+ wandb云端可视化使用 https://blog.csdn.net/q_xiami123/article/details/116937033
+ python配置库yacs https://blog.csdn.net/qq_41185868/article/details/103881451


## 待测试
固定non-iid度，调节算法超参数
1. 单独测试调度方案，调节信息量计算轮次
2. 单独测试非对称蒸馏方案，调节蒸馏的批次和轮次
3. 测试完整方案

固定算法超参数，调节non-iid度
1. 单独测试调度方案，调节hetero beta值和shards
2. 单独测试非对称蒸馏方案，调节hetero beta值和shards
3. 测试完整方案

### 待实现
1. 聚合方案出现问题
2. 聚合方案消除中心数据要求，用公式推导拟合数据分布情况


### 待优化
1. 中心端学习聚合参数更加细粒度，当前为节点一个数，后为和参数相同的规格 
2. 在新环境配置时候，需要手动创建logs/exps目录
3. 在新环境配置时候，需要手动创建logs/super目录
4. 调度和聚合过程都存在中心化数据假设，如何解决


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