# Application -12
# supervised machine learing 
# logistic regration
# There is one dataset which contains information about the passengers from Titanic
# This data set descibe multiple features about servived and non servived passenders

###########################################
# consider below characteristics of ML Application
# classifier - Logistic Regression
# DataSet    - Titanic Dataset
# Features   - Passenger id,Gender, Age, Fare, Class etc
# Labels     -
###################################################

import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def MyTitanicLogistic():
    # Step1: Load Data
    titanic_data = pd.read_csv('MyTitanicDataset.csv')
    
    print("First 5 entries from loaded dataset")
    print(titanic_data.head())
    
    print("Number of Passengers are: "+str(len(titanic_data)))
    
    # Step2: Analyze data
    print("Visualisation : Survived and non survived passanger")
    figure()
    target = "Survived"
    
    countplot(data = titanic_data, x = target).set_title("Survived and Non Survived passanger")
    show()
    
    print("Visualisation : Survived and non survived passanger based on the Gender")
    figure()
    target = "Survived"
    
    countplot(data = titanic_data, x = target, hue = "Sex").set_title("Survived and Non Survived Based on the Gender")
    show()
    
    print("Visualisation : Survived and non survived passengers based on the passenger class")
    figure()
    target = "Survived"
    
    countplot(data=titanic_data, x=target, hue="Pclass").set_title("Survived and non survived passangers based on the Passanger class")
    show()
    
    print("Visualisation : Survived and non survived Passanger based on the Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non survived passanger based on Age")
    show()
    
    print("Visualisation : Survived and non survived passanger based on the Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and non survived passanger based on Fare")
    show()
    
    # Step 3 : Data Cleaning
    titanic_data.drop("zero",axis =1, inplace=True)
    
    print("First 5 entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))
    
    print("Values of sex column")
    print(pd.get_dummies(titanic_data["Sex"]))
    
    print("Values of sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first=True)
    print(Sex.head(5))
    
    print("Values of pclass column after removing one field")
    Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print(Pclass.head(5))
    
    print("Values of data set after concatenating new columns")
    titanic_data = pd.concat([titanic_data,Sex,Pclass],axis=1)
    print(titanic_data.head(5))
    
    print("Values of data set after removing irrelevant columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head(5)) 
    
    x = titanic_data.drop("Survived",axis =1)
    y = titanic_data["Survived"]
    
    # Step 4 : Data Training
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.5)
    
    logmodel = LogisticRegression()
    logmodel.fit(xtrain,ytrain)
    
    # Step 5 : Data Testing
    predictions = logmodel.predict(xtest)
    
    # Step 6 : Calculate Accuracy
    
    print("Classification report of Logistic Regression is : ")
    print(classification_report(ytest,predictions))
    
    print("Confusion Matrix of Logistic Regression is : ")
    print(confusion_matrix(ytest,predictions))
    
    print("Accuracy of Logistic Regression is : ")
    print(accuracy_score(ytest,predictions))

def main():
    print("Suevised Machine Learning")
    print("Logistic Regression on Titanic Data set")
    MyTitanicLogistic()
if __name__ == "__main__":
    main()