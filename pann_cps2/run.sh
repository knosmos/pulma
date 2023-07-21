echo "=== BASELINE 25% PRETRAINED ==="
python train_preloaded.py --name nocps25_pretrained --shuffle --pretrained --percent 25 --batch 16
echo "=== BASELINE 12.5% PRETRAINED ==="
python train_preloaded.py --name nocps12_pretrained --shuffle --pretrained --percent 12.5 --batch 16
echo "=== BASELINE 50% PRETRAINED ==="
python train_preloaded.py --name nocps50_pretrained --shuffle --pretrained --percent 50 --batch 16
echo "=== BASELINE 6.25% PRETRAINED ==="
python train_preloaded.py --name nocps6_pretrained --shuffle --pretrained --percent 6.25 --batch 16
echo "=== CPS 25% PRETRAINED ==="
python train_preloaded.py --name cps25_pretrained --cps --shuffle --pretrained --percent 25 --batch 16
echo "=== CPS 12.5% PRETRAINED ==="
python train_preloaded.py --name cps12_pretrained --cps --shuffle --pretrained --percent 12.5 --batch 16
echo "=== CPS 50% PRETRAINED ==="
python train_preloaded.py --name cps50_pretrained --cps --shuffle --pretrained --percent 50 --batch 16
echo "=== CPS 6.25% PRETRAINED ==="
python train_preloaded.py --name cps6_pretrained --cps --shuffle --pretrained --percent 6.25 --batch 16