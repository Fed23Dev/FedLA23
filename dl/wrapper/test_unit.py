from matplotlib import pyplot as plt
from torch import optim
from torch.optim import SGD
from torchvision.models import resnet18

from dl.data.dataProvider import get_data_loader
from dl.model.model_util import create_model
from dl.wrapper.ExitDriver import ExitManager
from dl.wrapper.Wrapper import VWrapper
from dl.wrapper.optimizer.WarmUpCosinLR import WarmUPCosineLR
from dl.wrapper.optimizer.WarmUpStepLR import WarmUPStepLR
from env.running_env import args, file_repo


def exit_process(wrapper: VWrapper):
    exit_driver = ExitManager(wrapper)
    exit_driver.checkpoint_freeze()
    exit_driver.config_freeze()
    exit_driver.running_freeze()


def show_lr(net):
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # 打印当前学习率
    print(optimizer.state_dict()['param_groups'][0]['lr'])


def test_lr():
    net = resnet18(pretrained=False)
    optimizer = SGD(net.parameters(), lr=0.1)
    for i in range(5):
        optimizer.step()
        print("Optim:", optimizer.state_dict()['param_groups'][0]['lr'])
    epoch = 200
    lr = WarmUPStepLR(optimizer)
    x = [i + 1 for i in range(epoch)]
    y = []
    z = []

    for i in range(epoch):
        optimizer.step()
        lr.step()
        y.append(optimizer.state_dict()['param_groups'][0]['lr'])

    lr = WarmUPCosineLR(optimizer, T_max=epoch, lr_min=0.0)
    for i in range(epoch):
        optimizer.step()
        lr.step()
        z.append(optimizer.state_dict()['param_groups'][0]['lr'])

    print(y)
    print(z)

    # plt.plot(x, y)
    # plt.plot(x, z)
    # plt.show()


def test_running():
    model = create_model(args.model, num_classes=args.num_classes)
    dataloader = get_data_loader(args.dataset, data_type="train",
                                 batch_size=args.batch_size, shuffle=True)
    wrapper = VWrapper(model, dataloader, args.optim, args.scheduler, args.loss_func)
    wrapper.init_device(args.use_gpu, args.gpu_ids)
    wrapper.init_optim(args.learning_rate, args.momentum, args.weight_decay, args.nesterov)
    total_epoch = args.local_epoch * args.federal_round * args.active_workers if args.federal \
        else args.local_epoch
    wrapper.init_scheduler_loss(args.step_size, args.gamma, total_epoch, args.warm_steps, args.min_lr)
    if args.pre_train:
        wrapper.load_checkpoint(file_repo.model_path)

    wrapper.valid_performance(dataloader)


def main():
    test_running()
