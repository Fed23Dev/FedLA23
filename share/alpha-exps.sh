#!/usr/bin/env zsh

config_path='share/configs/alpha-exps.yml'

# super-parameter option
# 修改 line:9 超参数的取值 0.1 0.2 ~ 1.0
# 修改 line:12 超参数的字段 en_alpha: *.*/en_alpha

for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  echo "The value is: $alpha"
  sed -i "s/en_alpha: *.*/en_alpha: $alpha/g" $config_path

  for loop in 1 2 3
  do
    log_path="logs/super/alpha${alpha:2}.log$loop"
    nohup python main.py -y $config_path > $log_path 2>&1 && sleep 1
    wait
  done
done

# sed -i "s/en_alpha: *.*/en_alpha: $alpha/g" share/experiments/alpha-exps.yml
# sed -i 's/\r//g' share/experiments/alpha-exps.yml
# 为什么for里面不能嵌套while？

# nohup share/alpha-exps.sh &
# test dos
# set ff=unix