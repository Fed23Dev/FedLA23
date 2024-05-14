import argparse

import torch

from custom_path import auto_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml_path',
                        help='config file',
                        dest='config',
                        type=str,
                        default='e')
    parser.add_argument('-s', '--single_simulation',
                        action='store_true',
                        help='Federal simulation is False, the single is True.',
                        dest='env')
    parser.add_argument('-c', '--clean_files',
                        action='store_true',
                        help='Log files and others will be clean.',
                        dest='clean')
    init_arg = parser.parse_args()
    # !!!不要在调用这行命令前 引入running_env
    auto_config(init_arg.config)

    from dl.test_unit import main as dl_main
    from federal.test_unit import main as fl_main

    torch.backends.cudnn.benchmark = True

    if init_arg.env:
        dl_main()
    else:
        fl_main()

    # from utils.test_unit import main as statistics_main
    # # statistics_main()

    if init_arg.clean:
        from utils.Cleaner import FileCleaner
        cleaner = FileCleaner(remain_days=28, year=2023)
        cleaner.clear_files()
