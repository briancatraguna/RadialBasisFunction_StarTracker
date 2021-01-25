import pandas as pd
import nested_function as nf
import numpy as np
from math import degrees
import cv2

catalogue = pd.read_csv("Below_6.0_SAO.csv")

ra_list = list(catalogue['RA'])
de_list = list(catalogue['DE'])

images = []
for i in range(len(ra_list)):
    ra_list[i] = round(degrees(ra_list[i]),3)
    de_list[i] = round(degrees(de_list[i]),3)
    image = nf.create_star_image(ra_list[i],de_list[i],0)
    images.append(image)
    if i == 5:
        break

print(images)
for i in range(len(images)):
    image = images[i]
    height,width = image.shape
    y_center = height/2
    x_center = width/2
    x1 = int(x_center - 10)
    x2 = int(x_center + 10)
    y1 = int(y_center - 10)
    y2 = int(y_center + 10)
    cv2.rectangle(image,(x1,y1),(x2,y2),255,2)
    nf.displayImg(image)