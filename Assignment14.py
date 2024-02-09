import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load Data from CSV
data = pd.read_csv("PlayPredictor.csv")

# Step 2: Clean and Prepare Data
weather_encoder = LabelEncoder()
temperature_encoder = LabelEncoder()

data['Weather'] = weather_encoder.fit_transform(data['Weather'])
data['Temperature'] = temperature_encoder.fit_transform(data['Temperature'])

X = data[['Weather', 'Temperature']]
y = data['Play']

# Step 3: Train Data using KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Step 4: Test Data
new_data = pd.DataFrame({'Weather': ['Rainy'], 'Temperature': ['Mild']})
new_data['Weather'] = weather_encoder.transform(new_data['Weather'])
new_data['Temperature'] = temperature_encoder.transform(new_data['Temperature'])
prediction = knn.predict(new_data)
print(f'The prediction is: {prediction[0]}')

# Step 5: Calculate Accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
accuracies = []

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append((k, accuracy))

for k, accuracy in accuracies:
    print(f'K = {k}: Accuracy = {accuracy}')
