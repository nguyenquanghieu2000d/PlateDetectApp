import cv2
import matplotlib.pyplot as plt
from PIL import Image

image = cv2.imread("../image/22.jpg")
plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(plt_image)
im_pil.show()
