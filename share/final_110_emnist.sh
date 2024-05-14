#!/usr/bin/env zsh

config_path='share/configs/emnist-resnet110-final.yml'

for alpha in 20
do
  for loop in 1 2 3 4 5
  do
    log_path="logs/exps/resnet110.log$loop"
    nohup python main.py -y $config_path > $log_path 2>&1 && sleep 1
    wait
  done
done
