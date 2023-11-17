from dl.wrapper.Wrapper import VWrapper
from env.running_env import global_file_repo, args, global_container
from utils.objectIO import str_save, pickle_mkdir_save


def check(curt_key: str, one_key: str = None) -> bool:
    if one_key is None:
        return True
    else:
        return one_key == curt_key


def store_all(one_key: str = None):
    for key in global_container.keys:
        path, path_id = global_file_repo.new_seq(key)
        pickle_mkdir_save(global_container[key], path)


class ExitManager:
    def __init__(self, wrapper: VWrapper):
        self.wrapper = wrapper

    @staticmethod
    def config_freeze():
        config = args.get_snapshot()
        file, file_id = global_file_repo.new_exp(args.exp_name)
        str_save(config, file)
        return config

    def checkpoint_freeze(self, fixed: bool = True):
        name = str(args.model).split('.')[1]
        file, file_id = global_file_repo.new_checkpoint(name=name, fixed=fixed)
        self.wrapper.save_checkpoint(file)

    @staticmethod
    def running_freeze(indicator_key: str = None):
        store_all(indicator_key)
        paths = "\n".join(global_file_repo.reg_path)
        file, _ = global_file_repo.new_exp(f"{args.exp_name}_paths")
        str_save(paths, file)
