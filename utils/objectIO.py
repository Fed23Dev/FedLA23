import os
import pickle
import warnings


def dir_files(dir_path: str) -> list:
    all_files = []
    for filepath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            all_files.append(os.path.join(filepath, filename))
    return all_files


def create_path(f: str):
    dir_name = os.path.dirname(f)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)


def touch_file(f):
    create_path(f)
    fid = open(f, 'w')
    fid.close()


def remove_file(f_path: str):
    os.remove(f_path)


def remove_files(f_paths: list):
    for file in f_paths:
        os.remove(file)


def fetch_file_name(f_path: str):
    path_base, file = os.path.split(f_path)
    return file


def fetch_path_id(f_path: str) -> str:
    path_base, file = os.path.split(f_path)
    _file_name, file_postfix = os.path.splitext(file)
    return _file_name


# load obj not only wrapper: nn.Moudle
def pickle_mkdir_save(obj, f: str):
    create_path(f)
    # disabling warnings from torch.Tensor's reduce function. See issue: https://github.com/pytorch/pytorch/issues/38597
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(f, "wb") as opened_f:
            pickle.dump(obj, opened_f)
            opened_f.close()


def pickle_load(f):
    with open(f, "rb") as opened_f:
        obj = pickle.load(opened_f)
    return obj


def compare_obj(obj1, obj2) -> bool:
    return obj1 is obj2


def str_save(text: str, f: str):
    f = os.path.expanduser(f)
    with open(f, "w") as opened_f:
        opened_f.write(text)


if __name__ == '__main__':
    print("Nothing.")
