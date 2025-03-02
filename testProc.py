import torch
import torch.nn as nn
import time

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型定义
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
).to(device)

# 随机生成输入数据并将其传送到 GPU
input_data = torch.randn(32, 3, 224, 224).to(device)

# 执行一次前向传播，确保显存被分配
output = model(input_data)

# 获取并输出显存信息
allocated_memory = torch.cuda.memory_allocated()
max_allocated_memory = torch.cuda.max_memory_allocated()
print(f"Allocated Memory: {allocated_memory / (1024 ** 3):.4f} GB")
print(f"Max Allocated Memory: {max_allocated_memory / (1024 ** 3):.4f} GB")

# 持续输出显存占用，模拟训练过程
for epoch in range(5):
    start_time = time.time()
    # 进行一次前向传播和反向传播
    output = model(input_data)
    output.sum().backward()  # 反向传播
    print(f"Epoch {epoch+1} - Time: {time.time() - start_time:.4f}s")
    allocated_memory = torch.cuda.memory_allocated()
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"Allocated Memory: {allocated_memory / (1024 ** 3):.4f} GB")
    print(f"Max Allocated Memory: {max_allocated_memory / (1024 ** 3):.4f} GB")
    time.sleep(1)  # 模拟训练过程中的延时
