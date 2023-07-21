import os
from tqdm import tqdm
from collections import defaultdict

folder = "data/train/"

secs = defaultdict(float)
for i in tqdm(os.listdir(folder)):
    if i.endswith(".txt"):
        with open(folder + i, "r") as f:
            lines = f.readlines()
            labels = [None] * len(lines)
            for i, line in enumerate(lines):
                label, start, end = line.split()
                dur = float(end.split(":")[-1]) - float(start.split(":")[-1])
                secs[label] += dur

print(dict(secs))