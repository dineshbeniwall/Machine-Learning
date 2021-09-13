import numpy as np
import matplotlib.pyplot as plt


X = np.arange(-5, 5, 0.15)
Y = np.arange(-5, 5, 0.15)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)+X*0.1

plt.figure().add_subplot(projection='3d').plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
plt.show()
