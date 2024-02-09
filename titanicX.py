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
    # Step 1: Load Data
    titanic_data = pd.read_csv('MyTitanicDataset.csv')

    print("First 5 entries from the loaded dataset")
    print(titanic_data.head())

    print("Number of Passengers are: " + str(len(titanic_data)))

    # Step 2: Visualizations
    figure(figsize=(10, 6))
    sns.countplot(data=titanic_data, x="Survived", hue="Sex").set_title("Survived vs Non-survived Based on Gender")
    plt.show()

    figure(figsize=(10, 6))
    sns.countplot(data=titanic_data, x="Survived", hue="Pclass").set_title("Survived vs Non-survived Based on Passenger Class")
    plt.show()

    figure(figsize=(10, 6))
    sns.histplot(titanic_data, x="Age", hue="Survived", kde=True).set_title("Survived vs Non-survived Based on Age")
    plt.show()

    figure(figsize=(10, 6))
    sns.histplot(titanic_data, x="Fare", hue="Survived", kde=True).set_title("Survived vs Non-survived Based on Fare")
    plt.show()

    # Step 3: Data Cleaning (if required)
    # Handle missing values and feature engineering can be performed here

    # Step 4: Data Preparation
    # Drop unnecessary columns, handle missing values, and convert categorical variables to numerical if needed
    titanic_data.drop("unnecessary_column_name", axis=1, inplace=True)
    # Perform any necessary feature engineering, encoding categorical variables, or handling missing data

    X = titanic_data.drop("Survived", axis=1)
    y = titanic_data["Survived"]

    # Step 5: Data Splitting
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Model Training
    log_model = LogisticRegression()
    log_model.fit(x_train, y_train)

    # Step 7: Model Evaluation
    predictions = log_model.predict(x_test)

    # Step 8: Model Performance Metrics
    print("Classification report of Logistic Regression is: ")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix of Logistic Regression is:")
    print(confusion_matrix(y_test, predictions))

    print("Accuracy of Logistic Regression is:")
    print(accuracy_score(y_test, predictions))
    
MyTitanicLogistic()