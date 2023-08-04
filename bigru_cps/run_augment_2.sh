echo "== run_augment_2.sh =="

echo "== No CPS - 6.25% ==" >> augment_roll_train.txt
python train_augment.py --name nocps6_augment_gru_lr0.0001_roll --percent 6.25 --shuffle --pretrained --epochs 50 --lr 0.0001 >> augment_roll_train.txt
echo "== No CPS - 6.25% ==" >> augment_roll_test.txt
python test_perf_auc.py nocps6_augment_gru_lr0.0001_roll >> augment_roll_test.txt

echo "== No CPS, Reverse - 6.25% ==" >> augment_roll_train.txt
python train_augment.py --name nocps6_augment_gru_lr0.0001_roll_rev --percent 6.25 --shuffle --pretrained --epochs 50 --lr 0.0001 --reverse >> augment_roll_train.txt
echo "== No CPS, Reverse - 6.25% ==" >> augment_roll_test.txt
python test_perf_auc.py nocps6_augment_gru_lr0.0001_roll_rev >> augment_roll_test.txt

echo "== CPS - 6.25% ==" >> augment_roll_train.txt
python train_augment.py --name cps6_augment_gru_fromnocps_lr0.0001_roll --percent 6.25 --shuffle --pretrained --epochs 50 --lr 0.0001 --checkpoint model_nocps6_augment_gru_lr0.0001_roll_f1.pt --cps >> augment_roll_train.txt
echo "== CPS - 6.25% ==" >> augment_roll_test.txt
python test_perf_auc.py cps6_augment_gru_fromnocps_lr0.0001_roll >> augment_roll_test.txt

echo "== CPS, Reverse - 6.25% ==" >> augment_roll_train.txt
python train_augment.py --name cps6_augment_gru_fromnocps_lr0.0001_roll_rev --percent 6.25 --shuffle --pretrained --epochs 50 --lr 0.0001 --reverse --checkpoint model_nocps6_augment_gru_lr0.0001_roll_rev_f1.pt --cps >> augment_roll_train.txt
echo "== CPS, Reverse - 6.25% ==" >> augment_roll_test.txt
python test_perf_auc.py cps6_augment_gru_fromnocps_lr0.0001_roll_rev >> augment_roll_test.txt


echo "== No CPS - 25% ==" >> augment_roll_train.txt
python train_augment.py --name nocps25_augment_gru_lr0.0001_roll --percent 25 --shuffle --pretrained --epochs 50 --lr 0.0001 >> augment_roll_train.txt
echo "== No CPS - 25% ==" >> augment_roll_test.txt
python test_perf_auc.py nocps25_augment_gru_lr0.0001_roll >> augment_roll_test.txt

echo "== CPS - 25% ==" >> augment_roll_train.txt
python train_augment.py --name cps25_augment_gru_fromnocps_lr0.0001_roll --percent 25 --shuffle --pretrained --epochs 50 --lr 0.0001 --checkpoint model_nocps25_augment_gru_lr0.0001_roll_f1.pt --cps >> augment_roll_train.txt
echo "== CPS - 25% ==" >> augment_roll_test.txt
python test_perf_auc.py cps25_augment_gru_fromnocps_lr0.0001_roll >> augment_roll_test.txt



echo "== No CPS - 50% ==" >> augment_roll_train.txt
python train_augment.py --name nocps50_augment_gru_lr0.0001_roll --percent 50 --shuffle --pretrained --epochs 50 --lr 0.0001 >> augment_roll_train.txt
echo "== No CPS - 50% ==" >> augment_roll_test.txt
python test_perf_auc.py nocps50_augment_gru_lr0.0001_roll >> augment_roll_test.txt

echo "== CPS - 50% ==" >> augment_roll_train.txt
python train_augment.py --name cps50_augment_gru_fromnocps_lr0.0001_roll --percent 50 --shuffle --pretrained --epochs 50 --lr 0.0001 --checkpoint model_nocps50_augment_gru_lr0.0001_roll_f1.pt --cps >> augment_roll_train.txt
echo "== CPS - 50% ==" >> augment_roll_test.txt
python test_perf_auc.py cps50_augment_gru_fromnocps_lr0.0001_roll >> augment_roll_test.txt