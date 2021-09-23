import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "./transfusion.csv"
data = pd.read_csv(file_path, delimiter=",")

X = data[['Recency (months)', 'Frequency (times)',
          'Monetary (c.c. blood)', 'Time (months)']].to_numpy()
y = data['whether he/she donated blood in March 2007'].to_numpy()
epochs = 10
m, n = X.shape
theta = np.zeros((n+1, 1))
X = np.insert(X, 0, 1, axis=1)
n_miss_list = []
for epoch in range(epochs):
    n_miss = 0
    for idx, x_i in enumerate(X):
        a = np.dot(x_i.T, theta)
        if (np.sign(a)*y[idx] < 0):
            theta += (y[idx]*x_i)
            n_miss += 1
    n_miss_list.append(n_miss)
yPredict = np.sign(np.dot(X, theta))
'''
plt.plot(X, theta)
plt.show()
'''
# print(yPredict)
print(x_i)
