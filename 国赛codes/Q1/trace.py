import matplotlib.pyplot as plt
import numpy as np


r0 = 8.8
a = 0.275 / np.pi

plt.figure(figsize=(12, 12))
r = np.arange(8.8, 13, 0.0001)
theta = r / a
x1 = r * np.cos(theta)
y1 = r * np.sin(theta)
plt.plot(x1, y1, color='b')
plt.axis('equal')
plt.show()
