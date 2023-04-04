from env import yaml2args


def test_yaml2args():
    args = yaml2args.ArgRepo(r'share/cifar10-vgg16.yml')
    args.activate()
    print("here")
