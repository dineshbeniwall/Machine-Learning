import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import random


N = 100
X = []
Y = []
m = 5

for i in range(N):
    x = random.random()
    y = m*x+random.random()
    X.append(x)
    Y.append(y)

# plt.plot(X, Y, 'o')
X = np.reshape(X, (-1, 1))
# print(X)

# Split the data into training/testing sets
X_train = X[:-50]
# print(X_train)
X_test = X[-50:]
# print(X_test)
# Split the targets into training/testing sets
Y_train = Y[:-50]
Y_test = Y[-50:]


# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pre = regr.predict(X_test)
# The coefficients
print('Coefficients: \n', regr.coef_)

# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.plot(X_test, Y_pre, color='blue', linewidth=2)
plt.show()
