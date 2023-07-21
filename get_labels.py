import os
folder = "data/train"
labels = set()
for file in os.listdir(folder):
    if file.endswith("_label.txt"):
        with open(os.path.join(folder, file), "r") as f:
            lines = f.readlines()
            for line in lines:
                label, start, end = line.split()
                labels.add(label)
print(labels)