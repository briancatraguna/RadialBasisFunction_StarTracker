import pandas as pd
import nested_function as nf
import numpy as np
from math import degrees
import cv2

catalogue = pd.read_csv("Below_6.0_SAO.csv")

ra_list = list(catalogue['RA'])
de_list = list(catalogue['DE'])

for i in range(len(ra_list)):
    images = []
    ra_list[i] = round(degrees(ra_list[i]),3)
    de_list[i] = round(degrees(de_list[i]),3)
    for roll in range(0,360,10):
        image = nf.create_star_image(ra_list[i],de_list[i],roll)
        images.append(image)
    if i == 1:
        break

#VALIDATION
print(images)
for i in range(len(images)):
    image = images[i]
    height,width = image.shape
    y_center = height/2
    x_center = width/2
    x1 = int(x_center - 20)
    x2 = int(x_center + 20)
    y1 = int(y_center - 20)
    y2 = int(y_center + 20)
    cv2.rectangle(image,(x1,y1),(x2,y2),255,3)
    nf.displayImg(image)

