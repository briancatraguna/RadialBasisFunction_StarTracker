import pandas as pd
import nested_function as nf
import numpy as np
from math import degrees,sqrt,atan
import cv2

catalogue = pd.read_csv("Below_6.0_SAO.csv")

ra_list = list(catalogue['RA'])
de_list = list(catalogue['DE'])
star_id_list = list(catalogue['Star ID'])

image_test = nf.create_star_image(0,0,0)


def extract_rb_features(bin_increment,image,myu,f):
    """[This function extracts the radial basis features from a given star image]

    Args:
        bin_increment ([int]): [The bin increment is the delta theta for the histogram of features]
        image ([numpy array]): [The star image]
        myu ([float]): [length per pixel]
        f ([float]): [focal length]
    """
    #Get all the centroids
    image = image.astype('uint8')
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minThreshold = 50
    params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    params.minArea = 1
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)
    coord = []
    for index,keypoint in enumerate(keypoints):
        x_centralstar = int(round(keypoints[index].pt[0]))
        y_centralstar = int(round(keypoints[index].pt[1]))
        coord.append((x_centralstar,y_centralstar))

    bins = []
    for co in coord:
        x = co[0] - (image.shape[1]/2)
        y = (image.shape[0]/2) - co[1]
        pixel_distance_to_center = sqrt((x**2)+(y**2))
        angular_distance_to_center = round(atan((pixel_distance_to_center*myu)/f),3)
        

    return coord

coord = extract_rb_features(2,image_test)