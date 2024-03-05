#!/usr/bin/env zsh

# 设置默认关键字为"final"
keyword="final"

# 如果提供了命令行参数，则使用第一个参数作为关键字
if [ $# -eq 1 ]; then
  keyword=$1
fi

# 使用ps命令配合grep来获取包含keyword的进程，然后用grep -v grep排除掉包含"grep"的行
# 使用head -n 2只获取前两行
output=$(ps -ef | grep "$keyword" | grep -v grep | head -n 2)

# 检查output是否为空
if [ -z "$output" ]; then
  echo "No processes found with keyword '$keyword'."
  exit 0
fi

echo "Found processes:"
echo "$output"

# 提取PID并询问是否杀死进程
echo "$output" | awk '{print $2}' | while read pid; do
    # 打印整行信息，询问用户
    process_line=$(ps -ef | grep $pid | grep -v grep)
    echo "Process to kill: $process_line"
    read -p "Are you sure you want to kill PID $pid? (y/n): " confirmation
    if [ "$confirmation" = "y" ]; then
        echo "Killing PID: $pid"
        kill -9 $pid
    else
        echo "Skipping PID $pid"
    fi
done

