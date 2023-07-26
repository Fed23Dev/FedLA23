#!/usr/bin/env zsh

config_path='share/configs/alpha-exps.yml'

# super-parameter option
# 修改 line:9 超参数的取值 0.1 0.2 ~ 1.0
# 修改 line:12 超参数的字段 en_alpha: *.*/en_alpha

# for alpha in 5 10 15 20 30 40 50 100
for alpha in 5 20 40 60 80 100 120 150
#for alpha in 1 3 5 8 10 12 15 18 20

do
  echo "The value is: $alpha"
#  sed -i "s/logits_batch_limit: *.*/logits_batch_limit: $alpha/g" $config_path
#  sed -n '34p' $config_path

  sed -i "s/KD_BATCH: *.*/KD_BATCH: $alpha/g" $config_path
  sed -n '38p' $config_path

#  sed -i "s/KD_EPOCH: *.*/KD_EPOCH: $alpha/g" $config_path
#  sed -n '39p' $config_path

  for loop in 1 2 3 4 5
  do
    log_path="logs/super/alpha$alpha.log$loop"
    nohup python main.py -y $config_path > $log_path 2>&1 && sleep 1
    wait
  done
done

# nohup share/logits-exps.sh &
# test dos
# set ff=unix
