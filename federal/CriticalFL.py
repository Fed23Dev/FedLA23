import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy
import argparse
import random
import numpy as np

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

# 模型聚合函数

## CriticalFL ep
# 处于CLP时，取每个客户端梯度的前L(0<L<1)进行聚合
def aggregate_models_true(global_model, client_models, gradients, L, weights):
    for param in global_model.parameters():
        param.data = torch.zeros_like(param)
    global_dict = global_model.state_dict()
    total = [0 for _ in range(len(gradients[0]))]
    indices = []
    for gradient, model in zip(gradients, client_models):
        norms = torch.tensor([torch.norm(x) for x in gradient])
        _, indice = torch.topk(norms, round(len(gradient) * L))
        indice = [int(x) for x in indice]
        indices.append(indice)
    
    for index, weight in enumerate(weights):
        for i in range(len(gradients[0])):
            if i in indices[index]:
                total[i] += weight

    for index, weight in enumerate(weights):
        for i, k in enumerate(global_dict.keys()):
            if i not in indices[index]:
                continue
            global_dict[k] += weight / total[i] * model.state_dict()[k].float()
    
    global_model.load_state_dict(global_dict)
    return global_model
## CriticalFL ep

# 不处于CLP时，采用FedAvg聚合方式
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

## CriticalFL sp
# 判断是否处于CLP, True代表处于CLP， False代表不处于CLP
def check_clp(last_fgn, weights, gradients, eta, delta):
    now_fgn = cal_fgn(weights, gradients, eta)
    if (now_fgn - last_fgn) / last_fgn >= delta:
        return True, now_fgn
    return False, now_fgn

# 计算FGN
def cal_fgn(weights, gradients, eta):
    total = sum(weights)
    res = 0
    for weight, gradient in zip(weights, gradients):
        res = res + weight / total * -eta * (torch.norm(torch.tensor([torch.norm(x) for x in gradient])) ** 2)
    return res
## CriticalFL ep

# 客户端训练函数
def train_client(model, dataloader, criterion, optimizer, device, rounds):
    last_model = copy.deepcopy(model)

    #更新优化器的参数
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

    # 计算梯度
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    gra = [(param2 - param1) / lr / rounds for param1, param2 in zip(model.parameters(), last_model.parameters())]

    return total_loss / len(dataloader) / rounds, gra


# 客户端选择函数  这里把选取客户端比例的fraction参数更换为part_clients，方便后续part_clients的变换进行选取
def select_clients(num_clients, part_clients, client_models, client_optimizers, client_dataloaders, weights):
    selected_clients = random.sample(range(num_clients), part_clients)
    return [client_models[i] for i in selected_clients], \
           [client_optimizers[i] for i in selected_clients], \
           [client_dataloaders[i] for i in selected_clients], \
            [weights[i] for i in selected_clients]

# 获取每个客户端的权重
def get_weights(dataloaders):
    total = sum([len(dataloader) for dataloader in dataloaders])
    return [len(dataloader) / total for dataloader in dataloaders]

# 创建数据加载器
def create_loaders(args):
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # 使用Dirichlet分布划分数据集
    client_datasets = split_data_dirichlet(mnist_dataset, args.num_clients, args.dirichlet_alpha)
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
            proportions = np.array([p * (len(idx_j) < len(mnist_dataset) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    return [Subset(mnist_dataset, idx_j) for idx_j in idx_batch]

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='CriticalFL with PyTorch and MNIST')
    
    parser.add_argument('--L', type=float, default=0.5, help='Fraction of dimensions of the gradient to be transferred')
    parser.add_argument('--M', type=int, default=96, help='Most number of clients in federated learning')
    parser.add_argument('--m', type=int, default=32, help='Least number of clients in federated learning')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold of the FGN')

    parser.add_argument('--num_clients', type=int, default=128, help='Number of clients in federated learning')
    parser.add_argument('--part_clients', type=int, default=64, help='Number of clients participated in federated learning')
    parser.add_argument('--client_rounds', type=int, default=2, help='Number of local rounds for client training')
    parser.add_argument('--num_rounds', type=int, default=200, help='Number of federated learning rounds')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for SGD optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for client training')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, help='Concentration parameter for Dirichlet distribution')
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = create_loaders(args)
    weights = get_weights(dataloaders)

    # 创建全局模型和客户端模型
    global_model = SimpleCNN().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(args.num_clients)]
    optims = [torch.optim.SGD(model.parameters(), lr=args.lr) for model in client_models]
    criterion = nn.CrossEntropyLoss()
    all_loss = []
    # 训练和聚合
    last_fgn = 1
    for _ in range(args.num_rounds):
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Round", round + 1, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        selected_clients, selected_optims, selected_dataloaders, selected_weights = select_clients(
            args.num_clients, args.part_clients, client_models, optims, dataloaders, weights)
        round_loss = 0
        gradients = []
        for model, dataloader, optim in zip(selected_clients, selected_dataloaders, selected_optims):
            model.to(device)
            train_loss, gradient = train_client(model, dataloader, criterion, optim, device, args.client_rounds) #客户端训练 这里需要返回梯度值给后面进行CLP判断
            # print(f"Client Loss: {train_loss}")
            round_loss += train_loss 
            gradients.append(gradient)    
        round_loss /= args.part_clients
        all_loss.append(round_loss)
        print(f"Round {_ + 1}, Num of Clients {args.part_clients}, Loss: {round_loss}")

        ## CriticalFL sp
        # 判断是否处于CLP
        CLP, last_fgn = check_clp(last_fgn, selected_weights, gradients, args.lr, args.threshold)
        CLP = CLP or _ < 5
        # 如果处在CLP，则只传更新参数的部分，然后对part_clients进行调整
        if CLP:
            global_model = aggregate_models_true(global_model, selected_clients, gradients, args.L, selected_weights)
            args.part_clients = min(args.M, args.part_clients * 2)
        # 如果不处在，则跟FedAvg一样。但是仍需要对part_clients进行调整
        else:
            global_model = aggregate_models(global_model, selected_clients, selected_weights)
            args.part_clients = round(max(0.5 * args.part_clients, args.m))
        ## CriticalFL ep

        # 更新客户端模型
        client_models = [copy.deepcopy(global_model) for _ in client_models]


    # print("Final global model parameters:", global_model.state_dict())

if __name__ == "__main__":
    main()
