import pandas as pd

#Reading in Pandas Dataframe and Cleaning Data
excel_catalogue = pd.read_excel('SAO.xlsx')
tidy_catalogue = excel_catalogue.rename(columns = {'Unnamed: 0': 'Star ID', 'Unnamed: 1': 'RA', 'Unnamed: 2': 'DE', 'Unnamed: 3': "Magnitude"}, inplace=False)

#Filtering Magnitude
magnitude_filter = float(input("Input magnitude that you want to filter: "))
bool_filtered = tidy_catalogue['Magnitude']<=magnitude_filter
filtered_catalogue = tidy_catalogue[bool_filtered]

#Saving to CSV File
file_name = "Below_" + str(magnitude_filter) + "_SAO.csv"
filtered_catalogue.to_csv(file_name)