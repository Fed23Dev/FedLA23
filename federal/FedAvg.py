import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

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

# 客户端训练函数
def train_client(model, dataloader, criterion, optimizer, rounds):
    # 更新优化器的参数
    for param_group in optimizer.param_groups:
        param_group['params'] = list(model.parameters())
    model.train()
    total_loss = 0
    for _ in range(rounds):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(dataloader) / rounds

# 模型聚合函数
def aggregate_models(global_model, client_models, weights):
    total = sum(weights)
    for param in global_model.parameters():
        param.data = torch.zeros_like(param)
    global_dict = global_model.state_dict()
    for weight, model in zip(weights, client_models):
        for k in global_dict.keys():
            global_dict[k] += weight / total * model.state_dict()[k].float()
    global_model.load_state_dict(global_dict)
    return global_model

# 客户端选择函数
def select_clients(num_clients, fraction, client_models, client_optimizers, client_dataloaders, weights):
    selected_clients = random.sample(range(num_clients), int(num_clients * fraction))
    return [client_models[i] for i in selected_clients], \
           [client_optimizers[i] for i in selected_clients], \
           [client_dataloaders[i] for i in selected_clients], \
           [weights[i] for i in selected_clients]

# 获取客户端权重
def get_weights(dataloaders):
    total = sum([len(dataloader) for dataloader in dataloaders])
    return [len(dataloader) / total for dataloader in dataloaders]

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
            proportions = np.array([p * (len(idx_j) < len(mnist_dataset) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    return [Subset(mnist_dataset, idx_j) for idx_j in idx_batch]

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='FedAvg with PyTorch and MNIST')
    parser.add_argument('--num_clients', type=int, default=128, help='Number of clients in federated learning')
    parser.add_argument('--num_rounds', type=int, default=200, help='Number of federated learning rounds')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for SGD optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for client training')
    parser.add_argument('--client_rounds', type=int, default=2, help='Number of local rounds for client training')
    parser.add_argument('--client_fraction', type=float, default=0.5, help='Fraction of clients to be selected each round')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, help='Concentration parameter for Dirichlet distribution')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # 使用Dirichlet分布划分数据集
    client_datasets = split_data_dirichlet(mnist_dataset, args.num_clients, args.dirichlet_alpha)
    dataloaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=True) for dataset in client_datasets]
    weights = get_weights(dataloaders)

    # 创建全局模型和客户端模型
    global_model = SimpleCNN().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
    optims = [torch.optim.SGD(model.parameters(), lr=args.lr) for model in client_models]
    criterion = nn.CrossEntropyLoss()

    # 训练和聚合
    all_loss = []
    for round in range(args.num_rounds):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Round", round + 1, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        round_loss = 0.0
        selected_clients, selected_optims, selected_dataloaders, selected_weights = select_clients(
            args.num_clients, args.client_fraction, client_models, optims, dataloaders, weights)

        for model, dataloader, optim in zip(selected_clients, selected_dataloaders, selected_optims):
            model.to(device)
            train_loss = train_client(model, dataloader, criterion, optim, args.client_rounds)
            # print(f"Client Loss: {train_loss}")
            round_loss += train_loss
        print(f"Round {round + 1} average loss: {round_loss / len(selected_clients)}")
        global_model = aggregate_models(global_model, selected_clients, selected_weights)
        client_models = [copy.deepcopy(global_model) for _ in client_models]
        all_loss.append(round_loss / len(selected_clients))
    
    # print("Final global model parameters:", global_model.state_dict())

if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()

