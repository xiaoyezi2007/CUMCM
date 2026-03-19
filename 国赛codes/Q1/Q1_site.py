from sympy import *
import openpyxl

l_head = 286
l_body = 165
a = 27.5 / pi
r0 = 16 * 55
v0 = 100


def ans_site(t):
    ansr = []
    anssita = []
    anssite = []
    r = symbols('r')
    fr = r * sqrt(r * r + a * a) / 2 + a * a * ln(
        (r + sqrt(r * r + a * a)) / (r0 + sqrt(r0 * r0 + a * a))) / 2 - r0 * sqrt(
        r0 * r0 + a * a) / 2 + a * v0 * t
    ansr.append(nsolve(fr, 700))
    anssita.append(ansr[0] / a)
    x0 = ansr[0] * cos(anssita[0])
    y0 = ansr[0] * sin(anssita[0])
    anssite.append(x0.evalf())
    anssite.append(y0.evalf())
    fr = (r * cos(r / a) - x0) *(r * cos(r / a) - x0) + (r * sin(r / a) - y0) *(r * sin(r / a) - y0)- l_head *l_head
    ansr.append(nsolve(fr, ansr[0] + 2))
    anssita.append(ansr[1] / a)
    for i in range(1, 224):
        x0 = ansr[i] * cos(anssita[i])
        y0 = ansr[i] * sin(anssita[i])
        fr = (r * cos(r / a) - x0) *(r * cos(r / a) - x0) + (r * sin(r / a) - y0) *(r * sin(r / a) - y0) - l_body *l_body
        ansr.append(nsolve(fr, ansr[i] + 2))
        anssita.append(ansr[i + 1] / a)
        x = ansr[i] * cos(anssita[i])
        y = ansr[i] * sin(anssita[i])
        anssite.append(x.evalf())
        anssite.append(y.evalf())
    return anssite

print("ok")
wb = openpyxl.load_workbook('result1.xlsx')
sheet = wb['位置']
for t in range(0, 301):
    ans = ans_site(t)
    for i in range(448):
        tmp = ans[i] / 100
        sheet.cell(i + 2, t + 2).value = str(round(tmp, 6))
wb.save('result1.xlsx')
