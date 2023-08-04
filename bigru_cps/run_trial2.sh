python train_baseline.py --name ablation_cnn_2 --percent 12.5 --shuffle --epochs 50 --lr 0.0001 --seed 125
echo "ablation_cnn_2" >> trial2.txt
python test_perf_baseline.py ablation_cnn_2 >> trial2.txt

python train_deconv.py --name ablation_cnn_gru_pretrain_2 --percent 12.5 --shuffle --epochs 50 --lr 0.0001 --pretrained --seed 125
echo "ablation_cnn_gru_pretrain_2" >> trial2.txt
python test_perf_auc.py ablation_cnn_gru_pretrain_2 >> trial2.txt

for PERC in 12.5 25 50 6.25
do
    python train_deconv.py --name baseline4_${PERC}_2 --percent ${PERC} --shuffle --epochs 50 --lr 0.0001 --seed 125
    echo "baseline4_${PERC}_2" >> trial2.txt
    python test_perf_auc.py baseline4_${PERC}_2 >> trial2.txt
done