#!/usr/bin/env zsh

config_path='share/configs/cifar10-vgg16-final#.yml'
log_path='logs/exps/final_16_10.log'

if [ -f $log_path ]; then
  echo "Created file will be cleared."
  echo -n "" > $log_path
fi

nohup python main.py -y $config_path > $log_path 2>&1 &

# python -m pdb main.py -y 'share/configs/cifar10-vgg16-final#.yml' -c