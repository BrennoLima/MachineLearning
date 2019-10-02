# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Elbow method to discovery how many cluster we have to use
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
smaller = wcss[0]
for i in range(1,10):
    if wcss[i]*i < smaller:
        smaller = wcss[i]

nClusters = wcss.index(smaller) + 1

# Visualize Elbow method
plt.plot(range(1,11), wcss)
plt.scatter(nClusters, wcss[nClusters-1], c='red')
plt.title('Elbow method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()


# Apply the K-means method to train
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# Predict groups
y_kmeans = kmeans.fit_predict(X)

# Visualize 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroids')
plt.title('Clients')
plt.xlabel('Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()









