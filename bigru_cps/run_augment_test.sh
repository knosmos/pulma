echo "== No CPS - 6.25% F1 ==" >> augment_roll_test.txt
python test_perf_auc.py nocps6_augment_gru_lr0.0001_roll_f1 >> augment_roll_test.txt
echo "== No CPS, Reverse - 6.25%  F1==" >> augment_roll_test.txt
python test_perf_auc.py nocps6_augment_gru_lr0.0001_roll_rev_f1 >> augment_roll_test.txt
echo "== CPS - 6.25% F1 ==" >> augment_roll_test.txt
python test_perf_auc.py cps6_augment_gru_fromnocps_lr0.0001_roll_f1 >> augment_roll_test.txt
echo "== CPS, Reverse - 6.25% F1 ==" >> augment_roll_test.txt
python test_perf_auc.py cps6_augment_gru_fromnocps_lr0.0001_roll_rev_f1 >> augment_roll_test.txt
echo "== No CPS - 25% F1 ==" >> augment_roll_test.txt
python test_perf_auc.py nocps25_augment_gru_lr0.0001_roll_f1 >> augment_roll_test.txt
echo "== CPS - 25% F1 ==" >> augment_roll_test.txt
python test_perf_auc.py cps25_augment_gru_fromnocps_lr0.0001_roll_f1 >> augment_roll_test.txt
echo "== No CPS - 50% F1 ==" >> augment_roll_test.txt
python test_perf_auc.py nocps50_augment_gru_lr0.0001_roll_f1 >> augment_roll_test.txt
echo "== CPS - 50% F1 ==" >> augment_roll_test.txt
python test_perf_auc.py cps50_augment_gru_fromnocps_lr0.0001_roll_f1 >> augment_roll_test.txt