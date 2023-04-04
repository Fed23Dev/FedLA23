#!/usr/bin/env zsh

config_path='share/configs/cifar100-resnet110.yml'
log_path='logs/exps/final_110_100.log'

if [ -f $log_path ]; then
  echo "Created file will be cleared."
  echo -n "" > $log_path
fi

nohup python main.py -y $config_path > $log_path 2>&1 &

# python -m pdb main.py -y 'share/configs/cifar100-resnet110.yml' -c
