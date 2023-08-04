for SEED in 1 2 3 4
do
    for PERC in 6.25 12.5 25 50
    do
        echo "baseline2_${PERC}_${SEED}"
        echo "baseline2_${PERC}_${SEED}" >> baseline2_train.txt
        python train_deconv.py --name baseline2_${PERC}_${SEED} --percent ${PERC} --shuffle --lr 0.0001 --seed $SEED --epochs 50 >> baseline2_train.txt
        echo "baseline2_${PERC}_${SEED}" >> baseline2_test.txt
        python test_perf_auc.py baseline2_${PERC}_${SEED} >> baseline2_test.txt
    done
done