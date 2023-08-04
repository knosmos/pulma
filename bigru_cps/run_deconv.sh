echo "== Deconvolutional GRU =="
echo "== No CPS - 12.5% =="
python train_deconv.py --name nocps12_deconv_gru --percent 12.5 --shuffle --pretrained
echo "== No CPS - 50% =="
python train_deconv.py --name nocps50_deconv_gru --percent 50 --shuffle --pretrained
echo "== No CPS - 6.25% =="
python train_deconv.py --name nocps6_deconv_gru --percent 6.25 --shuffle --pretrained
echo "== No CPS - 25% - No Pretrained =="
python train_deconv.py --name nocps25_deconv_gru_scratch --percent 25 --shuffle
echo "== CPS - 6.25% =="
python train_deconv.py --name cps6_deconv_gru_wt1.0_warmup --percent 6.25 --shuffle --pretrained --cps --cps_weight 1.0 --warmup 2 --epochs 50
echo "== CPS - 50% =="
python train_deconv.py --name cps50_deconv_gru_wt1.0_warmup --percent 50 --shuffle --pretrained --cps --cps_weight 1.0 --warmup 2 --epochs 80
echo "== No CPS - 100% =="
python train_deconv.py --name nocps100_deconv_gru --percent 99.9 --shuffle --pretrained