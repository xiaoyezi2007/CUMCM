import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

l_head = 2.86
l_body = 1.65


def circle(xc, yc, r, start, end):
    phi1 = start
    phi2 = end
    dphi = (phi2 - phi1) / np.ceil(200 * np.pi * r * (phi2 - phi1))
    array = np.arange(phi1, phi2, dphi)
    array = np.append(array, array[-1] + dphi)
    return xc + r * np.cos(array), yc + r * np.sin(array)


def create_image(t):
    df = pd.read_excel('result1.xlsx', sheet_name='位置')
    column_data = df[str(t) + ' s']
    data_array = column_data.values.tolist()
    x = data_array[::2]
    y = data_array[1::2]
    xpoints = np.array(x)
    ypoints = np.array(y)
    plt.figure(figsize=(8, 8))
    X, Y = circle(x[0], y[0], l_head, 0, 2 * np.pi)
    plt.plot(X, Y, color='r')
    plt.plot(xpoints, ypoints)
    plt.show()


create_image(0)
