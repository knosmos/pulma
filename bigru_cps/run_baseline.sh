for PERC in 6.25 12.5 25 50
do
    for SEED in 1 2 3 4
    do
        echo "baseline_${PERC}_${SEED}"
        echo "baseline_${PERC}_${SEED}" >> baseline_train.txt
        python train_baseline.py --name baseline_${PERC}_${SEED} --percent ${PERC} --shuffle --seed $SEED --epochs 50 >> baseline_train.txt
        echo "baseline_${PERC}_${SEED}" >> baseline_test.txt
        python test_perf_baseline.py baseline_${PERC}_${SEED} >> baseline_test.txt
    done
done