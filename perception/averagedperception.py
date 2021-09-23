import numpy as np
import pandas as pd

file_path = "./transfusion.csv"
data = pd.read_csv(file_path, delimiter=",")

X = data[['Recency (months)', 'Frequency (times)',
          'Monetary (c.c. blood)', 'Time (months)']].to_numpy()
y = data['whether he/she donated blood in March 2007'].to_numpy()
epochs = 200
m, n = X.shape
theta = np.zeros(n+1)
X = np.insert(X, 0, 1, axis=1)
count = 0
weights = []
counts = []

for epoch in range(epochs):
    for idx, x_i in enumerate(X):
        a = np.dot(x_i.T, theta)
        if (np.sign(a)*y[idx] <= 0):
            theta += (y[idx]*x_i)
            weights.append(theta)
            counts.append(count)
            count = 1
        else:
            count += 1

yPredict = np.sign(np.dot(counts, np.dot(weights, X.T)))
print(X)
