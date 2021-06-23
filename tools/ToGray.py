import os
import cv2
from tqdm import tqdm

count = 0
DIR = "NhungEqua"
for i in tqdm(os.listdir(DIR)):
    img = cv2.imread(os.path.join(DIR, i), cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    cv2.imwrite(os.path.join(DIR, i), img)
