# LA的个人开发日志

## 开发日志
+ SCAFFOLD https://blog.csdn.net/Cyril_KI/article/details/123241764

可使用生态
+ wandb云端可视化使用 https://blog.csdn.net/q_xiami123/article/details/116937033
+ python配置库yacs https://blog.csdn.net/qq_41185868/article/details/103881451



## 待测试


### 待实现



## 待优化
1. cell.wrapper.run_model 进一步封装，可抽象分离出来
2. 中心端学习聚合参数更加细粒度，当前为节点一个数，后为和参数相同的规格
3. 中心端学习聚合参数超参数调节
4. 在新环境配置时候，需要手动创建logs/exps目录



## 难点
+ 调度算法初始点和更新方式有问题，需要更新
+ 聚合算法的更新走向不对，性能更差
+ 蒸馏算法还未验证

## 需求
+ 根据一个既定联邦学习算法，可以自由地更变损失计算方法，而不需要重复地写过多的代码
+ 我需要重写wrapper里面的损失计算函数的代码，step_run调用得到损失值
+ step_run要调用loss_compute，那参数只能通过他的上下文给，参数一般是由上层的联邦学习节点给的，如何传递
+ loss_compute内部的实现逻辑怎么写，由哪里重写
+ 总结：
  + loss_compute的参数如何传递，挂载到Wrapper上，还是由step_run间接传入
  + loss_compute内部的实现逻辑怎么写，自己拿到全局上下文，还是根据参数判断


## 未来的工作




## 博客对项目进行说明
### 快速熟悉项目进行二次开发
#### 架构
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