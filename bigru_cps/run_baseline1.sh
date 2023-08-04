for SEED in 2 3 4
do
    for PERC in 6.25 12.5 25 50
    do
        echo "baseline1_${PERC}_${SEED}"
        echo "baseline1_${PERC}_${SEED}" >> baseline1_train.txt
        python train_deconv.py --name baseline1_${PERC}_${SEED} --percent ${PERC} --shuffle --seed $SEED --epochs 50 >> baseline1_train.txt
        echo "baseline1_${PERC}_${SEED}" >> baseline1_test.txt
        python test_perf_auc.py baseline1_${PERC}_${SEED} >> baseline1_test.txt
    done
done