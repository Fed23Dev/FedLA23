import time

from env.running_env import global_logger, global_container


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        global_logger.info(f"Function {func.__name__} took {execution_time} seconds to execute.")
        global_container.flash(f'{func.__name__}-time', execution_time)
        return result
    return wrapper
