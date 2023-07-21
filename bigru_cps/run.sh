echo "=== nocps25_pretrained_gru ==="
python train_preloaded.py --name nocps25_pretrained_gru_2 --pretrained --percent 25 --shuffle --batch 16
echo "=== cps25_pretrained_gru ==="
python train_preloaded.py --name cps25_pretrained_gru --cps --pretrained --percent 25 --shuffle --batch 16
echo "=== nocps12_pretrained_gru ==="
python train_preloaded.py --name nocps12_pretrained_gru --pretrained --percent 12.5 --shuffle --batch 16
echo "=== cps12_pretrained_gru ==="
python train_preloaded.py --name cps12_pretrained_gru --cps --pretrained --percent 12.5 --shuffle --batch 16
echo "=== nocps6_pretrained_gru ==="
python train_preloaded.py --name nocps6_pretrained_gru --pretrained --percent 6.25 --shuffle --batch 16
echo "=== cps6_pretrained_gru ==="
python train_preloaded.py --name cps6_pretrained_gru --cps --pretrained --percent 6.25 --shuffle --batch 16