#!/bin/bash
for i in {1..5}
do
    echo "====== [$(date)] Starting run $i ======"
    python -m rl_core.self_train --pol --exp-name PoL --total-timesteps 500000 --exploration-fraction 0.2 --buffer-size 50000 --learning-starts 100 --size 25
    echo ""
done