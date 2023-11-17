#!/usr/bin/env zsh

config_path='share/configs/hyper-exps.yml' # hetero fmnist
# config_path='share/configs/noniid-hyper-exps.yml' # shards fmnist
# config_path='share/configs/task-hyper-exps.yml' # cifar10

# super-parameter option
# 修改 line:15 超参数的取值 0.1 0.2 ~ 1.0
# 修改 line:19 超参数的字段 logits_batch_limit: *.*/logits_batch_limit

# for alpha in 5 10 15 20 30 40 50 100
# for alpha in 5 20 40 60 80 100 120 150
# for alpha in 1 3 5 8 10 12 15 18 20
# for alpha in 1 2 3 4 5 6 7 8 9 10
# for alpha in 20
for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

do
#  echo "The value is: $alpha"
#  sed -i "s/logits_batch_limit: *.*/logits_batch_limit: $alpha/g" $config_path
#  sed -n '37p' $config_path

  echo "The value is: $alpha"
  sed -i "s/drag: *.*/drag: $alpha/g" $config_path
  sed -n '34p' $config_path

#  sed -i "s/KD_BATCH: *.*/KD_BATCH: $alpha/g" $config_path
#  sed -n '39p' $config_path

  for loop in 1 2 3 4 5
  do
    log_path="logs/super/alpha$alpha.log$loop"
    nohup python main.py -y $config_path > $log_path 2>&1 && sleep 1
    wait
  done
done

# sed -i "s/en_alpha: *.*/en_alpha: $alpha/g" share/experiments/hyper-exps.yml
# sed -i 's/\r//g' share/experiments/hyper-exps.yml
# 为什么for里面不能嵌套while？

# nohup share/hyper-exps.sh &
# test dos
# set ff=unix

# python -m pdb main.py -y 'share/configs/hyper-exps.yml'
