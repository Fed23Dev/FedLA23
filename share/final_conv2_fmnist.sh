#!/usr/bin/env zsh

config_path='share/configs/fmnist-conv2-final.yml'

do
  for loop in 1 2 3 4 5
  do
    log_path="logs/exps/conv2$loop.log"
    nohup python main.py -y $config_path > $log_path 2>&1 && sleep 1
    wait
  done
done
