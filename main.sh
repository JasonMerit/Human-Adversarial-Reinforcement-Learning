#!/bin/bash
# Set shared --exp-name
SIZE=25
EXP_NAME="PoLTrain"
DQN='DQN'
RIN='RainPer'
echo "========== $EXP_NAME =========="

# for i in {1..7}
# do
#     echo "====== [$(date)] Starting run $i ======" 
#     # python -m rl_core.clean_rainbow.train --exp-name PoLRain --pol --total-timesteps 100000

#     # python -m rl_core.self_train --pol --exp-name $EXP_NAME$DQN --total-timesteps 2000000 --size $SIZE --exploration-fraction 0.2 --buffer-size 50000 --learning-starts 100
#     # python -m rl_core.self_train --pol --exp-name $EXP_NAME$RIN --total-timesteps 2000000 --size $SIZE --exploration-fraction 0.2 --buffer-size 50000 --learning-starts 100 --rain
#     # python -m rl_core.self_train --pol --exp-name $EXP_NAME$RIN --total-timesteps 2000000 --size $SIZE --exploration-fraction 0.2 --buffer-size 50000 --learning-starts 100 --rain --per
#     python -m rl_core.clean_rainbow.train --pol --exp-name $EXP_NAME$DQN --total-timesteps 2000000 --size $SIZE --exploration-fraction 0.2 --buffer-size 50000 --learning-starts 100
#     # python -m rl_core.self_train --pol --exp-name KEK --total-timesteps 2000000 --size 5 --exploration-fraction 0.2 --buffer-size 50000 --learning-starts 100 --rain --per --debug
#     echo ""
# done

python -m rl_core.eval.pol_eval $EXP_NAME$DQN $SIZE
python -m rl_core.eval.pol_eval PoLDQN $SIZE
# python -m rl_core.eval.pol_eval $EXP_NAME$"Rain" $SIZE
# python -m rl_core.eval.pol_eval $EXP_NAME$RIN $SIZE
python on_complete.py
read -p "Press key to continue.. " -n1 -s


# Replicating same results with DQN from train instead of self_train

# DQN self_train 7 runs
# $SIZE=25
# python -m rl_core.self_train --pol --exp-name $EXP_NAME$DQN --size $SIZE --total-timesteps 2000000 --exploration-fraction 0.2 --buffer-size 50000 --learning-starts 100
# python -m rl_core.eval.pol_eval PoLDQN 25
# PoLDQN total steps avg: 587337

# DQN train 7 runs
# EXP_NAME="PoLTrain"
# python -m rl_core.clean_rainbow.train --pol --exp-name $EXP_NAME$DQN --total-timesteps 2000000 --size $SIZE --exploration-fraction 0.2 --buffer-size 50000 --learning-starts 100
