# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:43:26 2019

@author: 809438
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Dendogram method to find ideal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))

# Visualizing the dendogram
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Analyzing the dendogram, the number of ideal clusters is 5
from sklearn.cluster import AgglomerativeClustering

HC = AgglomerativeClustering(n_clusters=5, affinity='euclidean')
y_clustering = HC.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_clustering == 0, 0], X[y_clustering == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_clustering == 1, 0], X[y_clustering == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_clustering == 2, 0], X[y_clustering == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_clustering == 3, 0], X[y_clustering == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_clustering == 4, 0], X[y_clustering == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

















