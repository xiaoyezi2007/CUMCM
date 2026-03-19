import matplotlib.pyplot as plt
from sympy import *
import numpy as np
import pandas as pd


def create_image(t):
    df = pd.read_excel('result1.xlsx', sheet_name='位置')
    column_data = df[str(t) + ' s']
    data_array = column_data.values.tolist()
    x = data_array[::2]
    y = data_array[1::2]
    xi=[]
    yi=[]
    cosx0=(x[1]-x[0])/2.86
    cosy0=(y[1]-y[0])/2.86
    x1=x[0]-0.275*cosx0
    y1=y[0]-0.275*cosy0
    x0=x1+0.15*cosy0
    y0=y1-0.15*cosx0
    xi.append(x0)
    yi.append(y0)
    x0+=3.41*cosx0
    y0+=3.41*cosy0
    xi.append(x0)
    yi.append(y0)
    x0-=0.3*cosy0
    y0+=0.3*cosx0
    xi.append(x0)
    yi.append(y0)
    x0-=3.41*cosx0
    y0-=3.41*cosy0
    xi.append(x0)
    yi.append(y0)
    x0+=0.3*cosy0
    y0-=0.3*cosx0
    xi.append(x0)
    yi.append(y0)
    plt.figure(figsize=(8, 8))
    xpoints = np.array(x)
    ypoints = np.array(y)

    
    plt.plot(xpoints, ypoints)
    plt.plot(xi,yi)
    xi.clear()
    yi.clear()
    
    for i in range(1,100):
        cosx=(x[i+1]-x[i])/1.65
        cosy=(y[i+1]-y[i])/1.65
        x1=x[i]-0.275*cosx
        y1=y[i]-0.275*cosy
        x0=x1+0.15*cosy
        y0=y1-0.15*cosx
        xi.append(x0)
        yi.append(y0)
        x0+=2.20*cosx
        y0+=2.20*cosy
        xi.append(x0)
        yi.append(y0)
        x0-=0.3*cosy
        y0+=0.3*cosx
        xi.append(x0)
        yi.append(y0)
        x0-=2.20*cosx
        y0-=2.20*cosy
        xi.append(x0)
        yi.append(y0)
        x0+=0.3*cosy
        y0-=0.3*cosx
        xi.append(x0)
        yi.append(y0)
        plt.plot(xi,yi)
        xi.clear()
        yi.clear()
    plt.show()
create_image(50)