#!/bin/bash
SIZE=10
EXP_NAME="MCTS"
# PER="PER"
BASE="Benchmark"  # Comparison against
echo "========== $EXP_NAME =========="

for i in {0..0}
do
    echo "====== [$(date)] Starting run $i ======" 
    python -m rl_core.train --pol --exp-name $EXP_NAME --size $SIZE --mcts --debug
    echo ""
done

# python -m rl_core.eval.pol_eval $BASE $SIZE
python -m rl_core.eval.pol_eval $EXP_NAME $SIZE
python -m rl_core.eval.pol_eval $BASE $SIZE
python on_complete.py
read -p "Press key to continue.. " -n1 -s
