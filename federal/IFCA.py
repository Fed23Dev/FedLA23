# 使用 IFCA sp 注释特别标明和FedAvg差异化代码块
# code from: https://github.com/jichan3751/ifca/tree/main/femnist/ifca
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy
import argparse
import random
import numpy as np
from time import time


# 神经网络模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


## IFCA sp
# 模型聚合函数
def aggregate_models(global_model, gradients, indexes, lr, m):
    for gradient, index in zip(gradients, indexes):
        for param, grad in zip(global_model[index].parameters(), gradient):
            param.data.sub_(lr * grad / m)


# 客户端训练函数
def train_client(model, global_model, dataloader, criterion, optimizer):
    model = copy.deepcopy(global_model)

    # 更新优化器的参数
    for param_group in optimizer.param_groups:
        param_group['params'] = list(model.parameters())

    model.train()
    total_loss = 0
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 计算梯度。这里也可以看出，在梯度平均的方法下，客户端学习率对结果没有影响
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    gra = [(param2 - param1) / lr for param1, param2 in zip(model.parameters(), global_model.parameters())]
    return gra, total_loss / len(dataloader)


# group选择函数，找到loss值最小的group
def select_group(models, dataloaders, criterion):
    indexes = []
    for dataloader in dataloaders:
        loss = []
        for model in models:
            loss.append(0.0)
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss[-1] += criterion(outputs, labels).item()
        indexes.append(loss.index(min(loss)))
    return indexes


## IFCA ep

# 客户端选择函数
def select_clients(num_clients, fraction, client_models, client_optimizers, client_dataloaders):
    selected_clients = random.sample(range(num_clients), int(num_clients * fraction))
    return [client_models[i] for i in selected_clients], \
        [client_optimizers[i] for i in selected_clients], \
        [client_dataloaders[i] for i in selected_clients]


# 创建数据加载器
def create_loaders(args):
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # 使用Dirichlet分布划分数据集
    client_datasets = split_data_dirichlet(
        mnist_dataset, args.num_clients, args.dirichlet_alpha)
    return [DataLoader(dataset, batch_size=args.batch_size, shuffle=True) for dataset in client_datasets]


# 使用Dirichlet分布划分数据集
def split_data_dirichlet(mnist_dataset, num_clients, alpha):
    labels = mnist_dataset.targets.numpy()
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(10):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < len(mnist_dataset) / num_clients)
                                    for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) *
                           len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,
            idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    return [Subset(mnist_dataset, idx_j) for idx_j in idx_batch]


# 解析命令行参数
def parse_arguments():
    ## IFCA sp
    parser = argparse.ArgumentParser(description='IFCA with PyTorch and MNIST')

    # group数量
    parser.add_argument('--num_groups', type=int, default=3, help='Number of groups in IFCA')

    # 客户端学习率，这里使用的梯度平均，所以该值对结果不起作用
    parser.add_argument('--client_lr', type=float, default=0.01, help='Learning rate for client training')
    ## IFCA ep

    parser.add_argument('--num_clients', type=int, default=4, help='Number of clients in federated learning')
    parser.add_argument('--num_rounds', type=int, default=50, help='Number of federated learning rounds')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for server aggregation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for client training')
    parser.add_argument('--client_fraction', type=float, default=0.75,
                        help='Fraction of clients to be selected each round')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help='Concentration parameter for Dirichlet distribution')

    return parser.parse_args()


def main():
    args = parse_arguments()
    random.seed(int(time()))

    # 创建数据加载器
    dataloaders = create_loaders(args)

    ## IFCA sp
    # 创建全局模型，每个group分配一个model；创建客户端模型，每个工作机一个模型和一个优化器
    global_model = [SimpleCNN().to(device) for _ in range(args.num_groups)]
    client_models = [SimpleCNN().to(device) for _ in range(args.num_clients)]
    optims = [torch.optim.SGD(model.parameters(), lr=args.client_lr) for model in client_models]
    ## IFCA ep
    criterion = nn.CrossEntropyLoss()

    ## IFCA sp
    # 训练和聚合
    all_loss = []
    for round in range(args.num_rounds):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Round", round + 1, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # 选择工作机
        selceted_models, selected_dataloaders, selected_optims = select_clients(args.num_clients, args.client_fraction,
                                                                                client_models, dataloaders, optims)

        # 找到每个选中的工作机属于哪个group
        group_indexes = select_group(global_model, selected_dataloaders, criterion)

        gradients = []
        round_loss = 0.0
        for group_index, client_model, dataloader, optim in zip(group_indexes, selceted_models, selected_dataloaders,
                                                                selected_optims):
            gradient, train_loss = train_client(client_model, global_model[group_index], dataloader, criterion, optim)
            # print(f"Gradient: {gradient}")
            # print(f"Client Loss: {train_loss}")
            gradients.append(gradient)
            round_loss += train_loss
        print(f"Round {round + 1} average loss: {round_loss / len(group_indexes)}")
        all_loss.append(round_loss / len(group_indexes))
        aggregate_models(global_model, gradients, group_indexes, args.lr, args.num_clients)
    ## IFCA ep

    # for i, model in enumerate(global_model):
    #     print(f"Final global model parameters of group {i}:", model.state_dict())


if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
