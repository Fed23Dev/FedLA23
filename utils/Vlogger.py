import logging
import uuid

# 日志内容特征分析
# 自动删除超15天旧文件
# 日志id维护，保留旧id并加载

class VLogger():
    def __init__(self, file_path, sout = False) -> None:
        self.file_path = file_path
        self.sout = sout
        self.inner = logging.getLogger(self.log_id())

    def log_id(self) -> str:
        return str(uuid.uuid4())[:8]

    @property    
    def logger(self) -> logging.Logger:
        log_format = '[%(levelname)s] - %(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        file_handler = logging.FileHandler(self.file_path)
        file_handler.setFormatter(formatter)
        self.inner.addHandler(file_handler)

        if self.sout:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.inner.addHandler(stream_handler)

        self.inner.setLevel(logging.INFO)
        self.inner.propagate = False
        return self.inner
