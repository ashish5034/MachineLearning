# clustering using K-Mean algorithm
# in this case study we are using iris dataset with K-Mean algorithm from sklearn

# importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the iris dataset with pandas
dataset = pd.read_csv('iris.csv')
x = dataset.iloc[:,[0,1,2,3]].values

# finding the optimum number of cluster for K - Mean classification
from sklearn.cluster import KMeans
wcss = []

for i in range (1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# plotting the results onto a line graphh, allowing us to observe the elbow
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of cluster")
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

# Applying k Means to the dataset/creating the kmean classifier

kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Visualising the cluster
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s=100, c = 'red', label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s=100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s=100, c = 'green', label = 'Iris-Verginica')

# Plotting the centroids of the cluster

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='yellow', label = 'Centroids')

plt.legend()
plt.show()
