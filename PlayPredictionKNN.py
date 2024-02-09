import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def PlayPredictor(data_path):
    # step1: Load Data
    data = pd.read_csv(data_path, index_col=0)
    
    print("Size of Actual dataset", len(data))
    
    # Step2: clean, prepare and manipulate data
    feature_names = ['Weather','Temperature']
    
    print("Names of features: ",feature_names)
    
    Whether = data.Weather
    Temperature = data.Temperature
    play = data.Play
    
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    
    # converting string labels into number
    weather_encoded = le.fit_transform(Whether)
    print(weather_encoded)
    
    # converting string labels into number
    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)
    
    print(temp_encoded)
    
    # combining weather and temp into single listof tuples
    features = list(zip(weather_encoded,temp_encoded))
    
    # Step3 : Train data
    model = KNeighborsClassifier(n_neighbors=3)
    
    # Train the model using the training sets
    model.fit(features,label)
    
    # step4: test data
    predicted = model.predict([[0,2]])
    print(predicted)
        
def main():
    print("Machine Learning Application")
    
    print("Play Predictor application using K Nearest Neighbour algorithm")
    
    PlayPredictor("PlayPredictor.csv")
    
if __name__== "__main__":
    main()