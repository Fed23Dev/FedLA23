import re
import os
import numpy as np


class AccExtractor:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.acc_data = []
        self.acc_matrix = np.zeros(1)

    def extract_acc_data(self):
        # 获取匹配的文件列表并按照文件名排序
        files = sorted([filename for filename in os.listdir(self.log_dir)
                        if re.match(r".*\.log\d+$", filename)])

        for filename in files:
            file_path = os.path.join(self.log_dir, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
                # 从末尾开始遍历文件行
                for line in reversed(lines):
                    match = re.search(r"Loss: .* \| Acc: (\d+\.\d+)", line)
                    if match:
                        acc_value = float(match.group(1))
                        self.acc_data.append(acc_value)
                        break  # 找到匹配后终止循环

    def show_detail_rets(self):
        self.acc_matrix = np.array(self.acc_data).reshape(-1, 5)
        print(self.acc_matrix)

    def show_avg_rets(self):
        # 计算每行的平均值
        row_means = np.mean(self.acc_matrix, axis=1)

        # 打印每行的平均值
        for i, mean in enumerate(row_means):
            print(f"Row {i} mean: {mean}")
