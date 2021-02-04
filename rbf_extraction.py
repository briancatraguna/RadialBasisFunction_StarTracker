import pandas as pd
import nested_function as nf
import numpy as np
from math import degrees,sqrt,atan
import cv2

catalogue = pd.read_csv("Below_6.0_SAO.csv")

ra_list = list(catalogue['RA'])
de_list = list(catalogue['DE'])
star_id_list = list(catalogue['Star ID'])

def extract_rb_features(bin_increment,image,myu,f):
    """[This function extracts the radial basis features from a given star image and returns the bin feature vectors]

    Args:
        bin_increment ([int]): [The bin increment is the delta theta for the histogram of features (IN DEGREES)]
        image ([numpy array]): [The star image]
        myu ([float]): [length per pixel]
        f ([float]): [focal length]
    """
    #Defining some reusable variables to use
    half_length_pixel = image.shape[1]/2
    half_width_pixel = image.shape[0]/2
    FOVy_half = degrees(atan((half_width_pixel*myu)/f))

    #Initializing the bin list
    length_of_bin = FOVy_half//bin_increment
    bin_list = [0] * int(length_of_bin)

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
    #Iterating through all the stars present
    for index,keypoint in enumerate(keypoints):
        x_centralstar = int(round(keypoints[index].pt[0]))
        y_centralstar = int(round(keypoints[index].pt[1]))
        #Converting to origin-in-the-middle coordinates
        x = x_centralstar - half_length_pixel
        y = half_width_pixel - y_centralstar
        pixel_distance_to_center = sqrt((x**2)+(y**2))
        angular_distance_to_center = round(degrees(atan((pixel_distance_to_center*myu)/f)),3)
        if angular_distance_to_center > FOVy_half:
            continue
        lower_bound = 0
        upper_bound = bin_increment
        bin_index = 0
        #Evaluate which bin is this star in
        while upper_bound <= FOVy_half:
            if lower_bound <= angular_distance_to_center < upper_bound:
                bin_list[bin_index] += 1
            lower_bound += bin_increment
            upper_bound += bin_increment
            bin_index += 1

    return bin_list
    

feature_vector_dataset = {
    'Star ID'   : [],
    'Bin 1'     : [],
    'Bin 2'     : [],
    'Bin 3'     : [],
    'Bin 4'     : [],
    'Bin 5'     : [],
    'Bin 6'     : [],
    'Bin 7'     : [],
    'Bin 8'     : [],
    'Bin 9'     : [],
    'Bin 10'    : [],
    'Bin 11'    : [],
    'Bin 12'    : []
}

#Constants
bin_increment = 2
myu = 1.12*(10**-6)
f = 0.00304

col_list = ["Star ID","RA","DE","Magnitude"]
star_catalogue = pd.read_csv('Below_6.0_SAO.csv',usecols=col_list)

for i in range(len(star_id_list)):
    print("STAR {0} of {1}".format(i+1,len(star_id_list)))
    star_id = star_id_list[i]
    ra = degrees(ra_list[i])
    de = degrees(de_list[i])
    for roll in range(0,360,24):
        image = nf.create_star_image(ra,de,roll,1,1,0.3,star_catalogue=star_catalogue)
        features = extract_rb_features(bin_increment=bin_increment,image=image,myu=myu,f=f)
        feature_vector_dataset['Star ID'].append(star_id)
        print("Creating features for Star ID: {0} and Roll: {1}".format(star_id,roll))
        print("Features: {0}".format(features))
        for bin_number in range(len(features)):
            column_name = "Bin {0}".format(bin_number+1)
            feature_vector_dataset[column_name].append(features[bin_number])

feature_vector_dataframe = pd.DataFrame(
    feature_vector_dataset,
    columns=['Star ID','Bin 1','Bin 2','Bin 3','Bin 4','Bin 5','Bin 6','Bin 7','Bin 8','Bin 9','Bin 10','Bin 11','Bin 12'])

feature_vector_dataframe.to_csv('Without_Noise.csv',index=False)