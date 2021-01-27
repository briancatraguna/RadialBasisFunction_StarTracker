import pandas as pd
import nested_function as nf
import numpy as np
from math import degrees,sqrt
import cv2

catalogue = pd.read_csv("Below_6.0_SAO.csv")

ra_list = list(catalogue['RA'])
de_list = list(catalogue['DE'])
star_id_list = list(catalogue['Star ID'])

image_test = nf.create_star_image(0,0,0)


def extract_rb_features(bin_increment,image):
    """[This function extracts the radial basis features from a given star image]

    Args:
        bin_increment ([int]): [The bin increment is the delta theta for the histogram of features]
        image ([numpy array]): [The star image]
    """
    #Get all the centroids
    image = image.astype('uint8')

    #Set up the detector
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

    keypoints = detector.detect(image_test)
    coord = []
    for index,keypoint in enumerate(keypoints):
        x_centralstar = int(round(keypoints[index].pt[0]))
        y_centralstar = int(round(keypoints[index].pt[1]))
        coord.append((x_centralstar,y_centralstar))