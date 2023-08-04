echo "== Deconvolutional GRU Reversed =="
echo "== No CPS - 50% =="
python train_deconv.py --name nocps50_deconv_gru_rev --percent 50 --shuffle --pretrained
echo "== No CPS - 25% =="
python train_deconv.py --name nocps25_deconv_gru_rev --percent 25 --shuffle --pretrained --epochs 50
echo "== No CPS - 12.5% =="
python train_deconv.py --name nocps12_deconv_gru_rev --percent 12.5 --shuffle --pretrained --epochs 50
echo "== No CPS - 6.25% =="
python train_deconv.py --name nocps6_deconv_gru_rev --percent 6.25 --shuffle --pretrained --epochs 50
echo "== CPS - 50% =="
python train_deconv.py --name cps50_deconv_gru_fromnocps_rev --percent 50 --shuffle --pretrained --cps --cps_weight 1.0 --checkpoint model_nocps50_deconv_gru_rev.pt --epochs 50
echo "== CPS - 25% =="
python train_deconv.py --name cps25_deconv_gru_fromnocps_rev --percent 25 --shuffle --pretrained --cps --cps_weight 1.0 --checkpoint model_nocps25_deconv_gru_rev.pt --epochs 50
echo "== CPS - 12.5% =="
python train_deconv.py --name cps12_deconv_gru_fromnocps_rev --percent 12.5 --shuffle --pretrained --cps --cps_weight 1.0 --checkpoint model_nocps12_deconv_gru_rev.pt --epochs 50