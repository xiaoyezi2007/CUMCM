from sympy import *
import openpyxl

l_head = 286
l_body = 165
a = 27.5 / pi
r0 = 16 * 55
v0 = 100


def ans_verb(t):
    ansr = []
    anssita = []
    v = []
    v.append(1.000000)
    r = symbols('r')
    fr = r * sqrt(r * r + a * a) / 2 + a * a * ln(
        (r + sqrt(r * r + a * a)) / (r0 + sqrt(r0 * r0 + a * a))) / 2 - r0 * sqrt(
        r0 * r0 + a * a) / 2 + a * v0 * t
    ansr.append(nsolve(fr, 810))
    anssita.append(ansr[0] / a)
    x0 = ansr[0] * cos(anssita[0])
    y0 = ansr[0] * sin(anssita[0])
    fr = (r * cos(r / a) - x0) * (r * cos(r / a) - x0) + (r * sin(r / a) - y0) * (r * sin(r / a) - y0) - l_head * l_head
    ansr.append(nsolve(fr, ansr[0] + 2))
    anssita.append(ansr[1] / a)
    for i in range(1, 223):
        x0 = ansr[i] * cos(anssita[i])
        y0 = ansr[i] * sin(anssita[i])
        fr = (r * cos(r / a) - x0) * (r * cos(r / a) - x0) + (r * sin(r / a) - y0) * (
                r * sin(r / a) - y0) - l_body * l_body
        ansr.append(nsolve(fr, ansr[i] + 2))
        anssita.append(ansr[i + 1] / a)

    cosbn = (ansr[0] * ansr[0] + l_head * l_head - ansr[1] * ansr[1]) / (2 * ansr[0] * l_head)
    cosan = ansr[0] / sqrt(ansr[0] * ansr[0] + a * a)
    sinan = a / sqrt(ansr[0] * ansr[0] + a * a)
    cosfn1 = (ansr[1] * ansr[1] + l_head * l_head - ansr[0] * ansr[0]) / (2 * ansr[1] * l_head)
    cosan1 = ansr[1] / sqrt(ansr[1] * ansr[1] + a * a)
    sinan1 = a / sqrt(ansr[1] * ansr[1] + a * a)
    nxt = v[0] * (sqrt(1 - cosbn * cosbn) * cosan - cosbn * sinan) / (
            sqrt(1 - cosfn1 * cosfn1) * cosan1 + cosfn1 * sinan1)
    v.append(abs(nxt.evalf()))
    for i in range(1, 223):
        cosbn = (ansr[i] * ansr[i] + l_body * l_body - ansr[i + 1] * ansr[i + 1]) / (2 * ansr[i] * l_body)
        cosan = ansr[i] / sqrt(ansr[i] * ansr[i] + a * a)
        sinan = a / sqrt(ansr[i] * ansr[i] + a * a)
        cosfn1 = (ansr[i + 1] * ansr[i + 1] + l_body * l_body - ansr[i] * ansr[i]) / (2 * ansr[i + 1] * l_body)
        cosan1 = ansr[i + 1] / sqrt(ansr[i + 1] * ansr[i + 1] + a * a)
        sinan1 = a / sqrt(ansr[1 + 1] * ansr[i + 1] + a * a)
        nxt = v[i] * (sqrt(1 - cosbn * cosbn) * cosan - cosbn * sinan) / (
                sqrt(1 - cosfn1 * cosfn1) * cosan1 + cosfn1 * sinan1)
        v.append(abs(nxt.evalf()))
    return v


wb = openpyxl.load_workbook('result1.xlsx')
sheet = wb['速度']
for t in range(200, 301):
    print(t)
    ans = ans_verb(t)
    for i in range(224):
        sheet.cell(i + 2, t + 2).value = str(round(ans[i], 6))
wb.save('result1.xlsx')
