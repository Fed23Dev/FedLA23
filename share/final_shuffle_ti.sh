#!/usr/bin/env zsh

config_path='share/configs/ti-shuffle-final.yml'

for alpha in 20

do
  echo "The value is: $alpha"
  sed -i "s/logits_batch_limit: *.*/logits_batch_limit: $alpha/g" $config_path
  sed -n '37p' $config_path

  for loop in 1 2 3 4 5
  do
    log_path="logs/super/alpha$alpha.log$loop"
    nohup python main.py -y $config_path > $log_path 2>&1 && sleep 1
    wait
  done
done
