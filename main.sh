#!/bin/bash
SIZE=25
EXP_NAME="Buffer"
PER="PER"
# BASE="NewReplay"  # Comparison against
echo "========== $EXP_NAME =========="

for i in {0..19}
do
    echo "====== [$(date)] Starting run $i ======" 
    python -m rl_core.train --pol --exp-name $EXP_NAME --size $SIZE
    python -m rl_core.train --pol --exp-name $EXP_NAME$PER --size $SIZE --per
    # python -m rl_core.train --pol --exp-name KEK --size 5 --per --debug
    echo ""
done

# python -m rl_core.eval.pol_eval $BASE $SIZE
python -m rl_core.eval.pol_eval $EXP_NAME $SIZE
python -m rl_core.eval.pol_eval $EXP_NAME$PER $SIZE
python on_complete.py
read -p "Press key to continue.. " -n1 -s
