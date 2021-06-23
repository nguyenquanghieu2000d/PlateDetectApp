import os
from tqdm import tqdm

count = 10000
DIR = "./Hue"
for i in tqdm(os.listdir(DIR)):
    os.rename(os.path.join(DIR, i), os.path.join(DIR, "Hue." + str(count) + ".jpg"))
    count += 1
