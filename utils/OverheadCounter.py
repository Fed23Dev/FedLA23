import sys
import threading
import torch
import time
import subprocess

from env.running_env import global_container

class OverheadCounter:
    """
    计算内存占用
    interval: 每次记录内存占用的间隔时间（秒）
    duration: 记录内存占用的总时长（秒）
    """
    def __init__(self, interval=1, duration=10):
        self.interval = interval
        self.duration = duration
        self.stop_event = threading.Event()

    def log_memory_usage(self):
        allocated_memory = torch.cuda.memory_allocated()
        max_allocated_memory = torch.cuda.max_memory_allocated()
        global_container.flash('allocated_memory@GB', round(allocated_memory / (1024 ** 3), 4))
        global_container.flash('max_allocated_memory@GB', round(max_allocated_memory / (1024 ** 3), 4))

    def monitor_memory(self):
        start_time = time.time()
        while not self.stop_event.is_set() and time.time() - start_time < self.duration:
            self.log_memory_usage()
            time.sleep(self.interval)

    def start(self):
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_memory)
        self.monitor_thread.start()

    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()

    def calculate_memory_size(self, data, unit='bytes', downlink=False):
        """
        根据数据类型和单位计算内存占用
        :param downlink:
        :param data: 输入的数据（可以是基本数据类型、tensor 或其他类型）
        :param unit: 单位，可以是 'bytes', 'KB', 'MB', 'GB'
        :return: 该数据占用的内存大小
        """
        # 获取数据的内存大小（字节）
        if isinstance(data, torch.Tensor):
            # 对于 tensor 数据，返回其元素的大小乘以其元素数量
            num_elements = data.numel()
            element_size = data.element_size()  # 每个元素的字节数
            memory_size_bytes = num_elements * element_size
        else:
            # 对于其他数据类型，使用 sys.getsizeof() 来获取内存占用
            memory_size_bytes = sys.getsizeof(data)

        memory_size = memory_size_bytes
        # 转换为指定单位
        if unit == 'bytes':
            pass
        elif unit == 'KB':
            memory_size = memory_size_bytes / 1024
        elif unit == 'MB':
            memory_size = memory_size_bytes / (1024 ** 2)
        elif unit == 'GB':
            memory_size = memory_size_bytes / (1024 ** 3)
        else:
            raise ValueError(f"Unsupported unit: {unit}")

        if downlink:
            global_container.flash('DownObject@'+unit, memory_size)
        else:
            global_container.flash('UpObject@'+unit, memory_size)

    def get_system_gpu_memory(self):
        """
        获取系统级别的 GPU 显存使用情况
        """
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE)
        gpu_info = result.stdout.decode('utf-8').strip().split('\n')

        gpu_memory = []
        for info in gpu_info:
            used, free, total = map(int, info.split(','))
            gpu_memory.append({
                'used_memory': used,
                'free_memory': free,
                'total_memory': total,
            })

        return gpu_memory
