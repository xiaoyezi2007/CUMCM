import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as ani

fig = plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.grid()
ax.set_xlim(-13, 13)
ax.set_ylim(-13, 13)
ln1, = ax.plot([], [], '-', lw=2)
line = []
line.append(ln1)
time_template = 'a = %.1f cm/rad'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
for i in range(1, 101):
    ln, = ax.plot([], [], '-', lw=2)
    line.append(ln)
df = pd.read_excel('result1(for_r=450).xlsx', sheet_name='位置')


def update(j):
    column_data = df[str(j) + ' s']
    data_array = column_data.values.tolist()
    x = data_array[::2]
    y = data_array[1::2]
    xi = []
    yi = []
    xpoints = np.array(x)
    ypoints = np.array(y)
    line[0].set_data(xpoints, ypoints)

    xi.clear()
    yi.clear()

    for i in range(1, 100):
        cosx = (x[i + 1] - x[i]) / 1.65
        cosy = (y[i + 1] - y[i]) / 1.65
        x1 = x[i] - 0.275 * cosx
        y1 = y[i] - 0.275 * cosy
        x0 = x1 + 0.15 * cosy
        y0 = y1 - 0.15 * cosx
        xi.append(x0)
        yi.append(y0)
        x0 += 2.20 * cosx
        y0 += 2.20 * cosy
        xi.append(x0)
        yi.append(y0)
        x0 -= 0.3 * cosy
        y0 += 0.3 * cosx
        xi.append(x0)
        yi.append(y0)
        x0 -= 2.20 * cosx
        y0 -= 2.20 * cosy
        xi.append(x0)
        yi.append(y0)
        x0 += 0.3 * cosy
        y0 -= 0.3 * cosx
        xi.append(x0)
        yi.append(y0)
        xpoints = np.array(xi)
        ypoints = np.array(yi)
        line[i].set_data(xpoints, ypoints)
        xi.clear()
        yi.clear()
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
    xpoints = np.array(xi)
    ypoints = np.array(yi)
    line[i].set_data(xpoints, ypoints)
    xi.clear()
    yi.clear()
    time_text.set_text(time_template %(2+j*0.1))
    return line, time_text


animation = ani.FuncAnimation(fig, update, range(50,65), interval=1000)

plt.show()
