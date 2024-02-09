# Required Python Packages

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# File Paths

INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"

# Headers

HEADERS =["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]

# Function Name: read_data
# Description: Read the data into Pandas dataframe
# input : path of csv file
# output : Gives the data
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def read_data(path):
    data = pd.read_csv(path)
    return data

# Function Name: get_headers
# Description: dataset headers
# input : dataset
# output : return the header
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def get_headers(dataset):
    return dataset.columns.values

# Function Name: add_headers
# Description: add the headers to the dataset
# input : dataset
# output : updated header
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def add_headers(dataset,headers):
    dataset.columns = headers
    return dataset

# Function Name: data_file_to_csv
# input : Nothing
# output : write the data to csv
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def data_file_to_csv():
    # Headers
    headers = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]
    # load the dataset into pandas data frame
    dataset = read_data(INPUT_PATH)
    # Add the headers to the loaded dataset
    dataset = add_headers(dataset,headers)
    # save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("fILE SAVED...!")
    
# Function Name: split_dataset
# Description: split the dataset with train percentage
# input : dataset with related information
# output : dataset after splitting 
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header], train_size=train_percentage)
    return train_x, test_x, train_y, test_y

# Function Name: handel_missing_values
# Description: filter missing values from dataset
# input : dataset with missing values
# output : dataset with remocing missing values
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def handel_missing_values(dataset,missing_values_header,missing_label):
    return dataset[dataset[missing_values_header]!=missing_label]

# Function Name: random_forest_classifier
# Description: to train the random forest classifier with features and target data
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def random_forest_classifier(features,target):
    clf = RandomForestClassifier()
    clf.fit(features,target)
    return clf

# Function Name: dataset_statics
# Description: basic statics of the  dataset
# input : dataset
# output : desciption of dataset
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def dataset_statics(dataset):
    print(dataset.describe())
    
# Function Name: main
# Description: main function from where execution starts
# Author: Ashish Kishor Mahamuni
# date =29/10/2023

def main():
    # load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    # get basic statistics of the loaded dataset
    dataset_statics(dataset)
    
    # filter missing values
    dataset = handel_missing_values(dataset, HEADERS[6],'?')
    train_x,test_x,train_y,test_y = split_dataset(dataset,0.7,HEADERS[1:-1],HEADERS[-1])
    
    # train and test dataset size details
    print("Train_x Shape::",train_x.shape)
    print("Train_y Shape::",train_y.shape)
    print("Test_x Shape::",test_x.shape)
    print("Test_y Shape::",test_y.shape)
    
    # create random forest classifier instance
    trained_model = random_forest_classifier(train_x,train_y)
    print("Trained Model::",trained_model)
    predictions = trained_model.predict(test_x)
    
    for i in range(len(test_y)):
        print("Actual outcome::{} and Predicted Outcome::{}".format(list(test_y)[i], predictions[i]))
        
    print("Train Accuracy::", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy::", accuracy_score(test_y, predictions))
    print("Confusion Matrix:", confusion_matrix(test_y, predictions))
        
# Application starter

if __name__ == "__main__":
    main()