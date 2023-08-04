for PERC in 50 25 12.5 6.25
do
    echo "proposed_${PERC}_1_pretrain"
    echo "proposed_${PERC}_1_pretrain" >> proposed_train.txt
    python train_deconv.py --name proposed_${PERC}_1_pretrain --percent ${PERC} --pretrained --shuffle --lr 0.0001 --seed 50 --epochs 50 >> proposed_train.txt
    echo "proposed_${PERC}_1" >> proposed_train.txt
    python train_augment.py --name proposed_${PERC}_1 --percent ${PERC} --cps --shuffle --lr 0.0001 --seed 50 --epochs 50 --checkpoint models/model_proposed_${PERC}_1_pretrain.pt >> proposed_train.txt
    echo "proposed_${PERC}_1" >> proposed_test.txt
    python test_perf_auc.py proposed_${PERC}_1 >> proposed_test.txt
done