from sympy import *
import numpy as np
import matplotlib.pyplot as plt

eps = 1e-12
l_head = 286
l_body = 165


def ans_site(a, r0):
    ansr = []
    anssita = []
    anssite = []
    r = symbols('r')
    fr = r * sqrt(r * r + a * a) / 2 + a * a * ln(
        (r + sqrt(r * r + a * a)) / (r0 + sqrt(r0 * r0 + a * a))) / 2 - r0 * sqrt(
        r0 * r0 + a * a) / 2
    ansr.append(nsolve(fr, 700))
    anssita.append(ansr[0] / a)
    x0 = ansr[0] * cos(anssita[0])
    y0 = ansr[0] * sin(anssita[0])
    anssite.append([x0.evalf(), y0.evalf()])
    fr = (r * cos(r / a) - x0) * (r * cos(r / a) - x0) + (r * sin(r / a) - y0) * (r * sin(r / a) - y0) - l_head * l_head
    ansr.append(nsolve(fr, ansr[0] + 2))
    anssita.append(ansr[1] / a)
    for i in range(1, 224):
        x0 = ansr[i] * cos(anssita[i])
        y0 = ansr[i] * sin(anssita[i])
        fr = (r * cos(r / a) - x0) * (r * cos(r / a) - x0) + (r * sin(r / a) - y0) * (
                r * sin(r / a) - y0) - l_body * l_body
        ansr.append(nsolve(fr, ansr[i] + 2))
        anssita.append(ansr[i + 1] / a)
        x = ansr[i] * cos(anssita[i])
        y = ansr[i] * sin(anssita[i])
        anssite.append([x.evalf(), y.evalf()])
    return anssite


def distance(x0, y0, x1, y1, x2, y2):
    dis = abs(x0 * (y2 - y1) + y0 * (x1 - x2) + (x2 * y1 - x1 * y2)) / sqrt(
        (y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2))
    return dis


r = np.arange(462.44, 462.46, 0.002)
a = np.arange(7.158871, 7.1588715, 0.00000005)
for i in a:
    print("a="+str(i),end=':')
    for j in r:
        ans = ans_site(i, j)
        cosx0 = (ans[1][0] - ans[0][0]) / 2.86
        cosy0 = (ans[1][1] - ans[0][1]) / 2.86
        x1 = ans[0][0] - 0.275 * cosx0
        y1 = ans[0][1] - 0.275 * cosy0
        x0 = x1 + 0.15 * cosy0
        y0 = y1 - 0.15 * cosx0
        if distance(x0, y0, ans[17][0], ans[17][1], ans[18][0], ans[18][1]) > 15:
            print(0,end=' ')
        else:
            print(1,end=' ')
    print()
