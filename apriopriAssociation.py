# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:39:51 2019

@author: 809438
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# The model needs the dataset as a list of lists, therefore, it's necessary to cast the dataset
transactions = []
aux = []
for i in range(0, 7501):
    if len(aux): 
        transactions.append(aux)
    aux = []
    for j in range(0,20):
        aux.append(str(dataset.values[i,j]))
    
# Fitting
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizing


# Arguments: rules from apriori method, number of associations
def visualize(rules, n):
    results_list = []
    results = list(rules)
    for i in range(0, len(results)):
        results_list.append(str(results[i][2]))
    
    if len(results_list) < n:
        n = len(results_list)
    products = ''  
    for i in range(0, n):
            for j in range(0, len(results_list[i])):
                if results_list[i][j] == '{':
                    while True:
                        j += 1
                        if results_list[i][j] == '}':
                            products += ' '
                            break
                        if results_list[i][j] == ',': 
                            j+=1
                        products += results_list[i][j]
            products += ': ' + results_list[i][results_list[i].find('confidence')+13:results_list[i].find('confidence')+15] + '%\n'
    products = products.replace("'nan'", '')  
    return products

print(visualize(rules, 10))
    
            













