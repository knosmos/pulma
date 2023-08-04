for PERC in 50 25 12.5 6.25
do
    echo "proposed_${PERC}_2_pretrain"

    echo "proposed_${PERC}_2_pretrain" >> proposed_train_2.txt
    python train_augment.py --name proposed_${PERC}_2_pretrain --percent ${PERC} --pretrained --shuffle --lr 0.0001 --seed 125 --epochs 50 >> proposed_train_2.txt
    echo "proposed_${PERC}_2_pretrain" >> proposed_test_2.txt
    python test_perf_auc.py proposed_${PERC}_2_pretrain >> proposed_test_2.txt

    echo "proposed_${PERC}_2"

    echo "proposed_${PERC}_2" >> proposed_train_2.txt
    python train_augment.py --name proposed_${PERC}_2 --augment --percent ${PERC} --cps --shuffle --lr 0.0001 --seed 125 --epochs 50 --checkpoint models/model_proposed_${PERC}_2_pretrain.pt >> proposed_train_2.txt
    echo "proposed_${PERC}_2" >> proposed_test_2.txt
    python test_perf_auc.py proposed_${PERC}_2 >> proposed_test_2.txt
done