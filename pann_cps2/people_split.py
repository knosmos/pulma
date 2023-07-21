import glob
import datetime
import numpy as np

people = []
curr_person = []
last_date = False

threshold = 47 # minutes

for i, fname in enumerate(sorted(glob.glob("../data/train/steth_*.wav"))):
    date = datetime.datetime.strptime(fname.split("/")[-1], "steth_%Y%m%d_%H_%M_%S.wav")
    if last_date:
        diff = (date - last_date).total_seconds() / 60
        if diff > threshold:
            people.append(np.array(curr_person))
            curr_person = []
    curr_person.append(i)
    last_date = date
people.append(np.array(curr_person))