echo "== No CPS - 12.5% =="
python train_augment.py --name nocps12_augment_gru_crop --percent 12.5 --shuffle --pretrained --epochs 50 --lr 0.0001
echo "== No CPS - 50% =="
python train_augment.py --name nocps50_augment_gru_crop --percent 50 --shuffle --pretrained --epochs 50 --lr 0.0001
echo "== No CPS - 6.25% =="
python train_augment.py --name nocps6_augment_gru_crop --percent 6.25 --shuffle --pretrained --epochs 50 --lr 0.0001
echo "== No CPS - 25% =="
python train_augment.py --name nocps25_augment_gru_crop --percent 25 --shuffle --pretrained --epochs 50 --lr 0.0001
echo "Testing..."
python test_perf_auc.py nocps50_augment_gru_crop >> augment_gru_crop_results.txt
python test_perf_auc.py nocps25_augment_gru_crop >> augment_gru_crop_results.txt
python test_perf_auc.py nocps12_augment_gru_crop >> augment_gru_crop_results.txt
python test_perf_auc.py nocps6_augment_gru_crop >> augment_gru_crop_results.txt
echo "== CPS - 12.5% =="
python train_augment.py --name cps12_augment_gru_fromnocps_crop --percent 12.5 --shuffle --pretrained --cps --cps_weight 1.0 --checkpoint model_nocps12_augment_gru_crop.pt --epochs 50 --lr 0.0001
echo "== CPS - 50% =="
python train_augment.py --name cps50_augment_gru_fromnocps_crop --percent 50 --shuffle --pretrained --cps --cps_weight 1.0 --checkpoint model_nocps50_augment_gru_crop.pt --epochs 50 --lr 0.0001
echo "== CPS - 6.25% =="
python train_augment.py --name cps6_augment_gru_fromnocps_crop --percent 6.25 --shuffle --pretrained --cps --cps_weight 1.0 --checkpoint model_nocps6_augment_gru_crop.pt --epochs 50 --lr 0.0001
echo "== CPS - 25% =="
python train_augment.py --name cps25_augment_gru_fromnocps_crop --percent 25 --shuffle --pretrained --cps --cps_weight 1.0 --checkpoint model_nocps25_augment_gru_crop.pt --epochs 50 --lr 0.0001