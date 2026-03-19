from sympy import *
import openpyxl
import numpy as np
import math
l_head = 286
l_body = 165
r0 = 450
a = 85 / np.pi
v0 = 100
theta0 = r0 / a
a0 = -np.arctan((np.sin(r0 / a) + (r0 / a) * np.cos(r0 / a)) / (np.cos(r0 / a) - (r0 / a) * np.sin(r0 / a)))
b0 = np.arctan(np.tan(r0 / a))
r1 = 2 * math.sqrt(r0 * r0 + a * a) / 3
r2 = math.sqrt(r0 * r0 + a * a) / 3
x1 = r0 * np.cos(theta0) + r1 * np.sin(a0)
y1 = r0 * np.sin(theta0) + r1 * np.cos(a0)  # 大圆心
x2 = -r0 * np.cos(theta0) - r2 * np.sin(a0)
y2 = -r0 * np.sin(theta0) - r2 * np.cos(a0)  # 小圆心
xc = -r0 * np.cos(theta0) / 3
yc = -r0 * np.sin(theta0) / 3  # 两圆交点
eps = 0.1
def archway_length(xc, yc, x1, y1, x2, y2, r):
    (x1 - xc) * (x2 - xc) + (y1 - yc) * (y2 - yc)
    theta = acos(((x1 - xc) * (x2 - xc) + (y1 - yc) * (y2 - yc)) / r / r)
    return r * theta
len1 = archway_length(x1, y1, r0 * np.cos(theta0), r0 * np.sin(theta0), xc, yc, r1)
len2 = archway_length(x2, y2, -r0 * np.cos(theta0), -r0 * np.sin(theta0), xc, yc, r2)

def location(t):
    if t < 0:
        s = v0 * t
        r = symbols('r')
        fr = r * sqrt(r * r + a * a) / 2 + a * a * ln(
            (r + sqrt(r * r + a * a)) / (r0 + sqrt(r0 * r0 + a * a))) / 2 - r0 * sqrt(
            r0 * r0 + a * a) / 2 + a * s
        r = nsolve(fr, r0 + 2)
        x = r * cos(r / a)
        y = r * sin(r / a)
        return [x, y]
    elif v0 * t <= len1:
        if r0 * np.sin(theta0) > y1:
            a1 = np.arccos((r0 * np.cos(theta0) - x1) / r1)
        else:
            a1 = 2 * np.pi - np.arccos((r0 * np.cos(theta0) - x1) / r1)
        theta = a1 - v0 * t / r1
        x = x1 + r1 * cos(theta)
        y = y1 + r1 * sin(theta)
        return [x, y]
    elif v0 * t <= len1 + len2:
        if -r0 * np.sin(theta0) / 3 > y2:
            a1 = np.arccos((-r0 * np.cos(theta0) / 3 - x2) / r2)
        else:
            a1 = 2 * np.pi - np.arccos((-r0 * np.cos(theta0) / 3 - x2) / r2)
        theta = a1 + (v0 * t -len1) / r2
        x = x2 + r2 * cos(theta)
        y = y2 + r2 * sin(theta)
        return [x, y]
    else:
        s = v0 * t - len1 - len2
        r = symbols('r')
        fr = r * sqrt(r * r + a * a) / 2 + a * a * ln(
            (r + sqrt(r * r + a * a)) / (r0 + sqrt(r0 * r0 + a * a))) / 2 - r0 * sqrt(
            r0 * r0 + a * a) / 2 - a * s
        r = nsolve(fr, r0 + 2)
        x = r * cos(r / a)
        y = r * sin(r / a)
        return [-x, -y]

def ans_site(t):
    ansr = []
    anssita = []
    anssite = []
    if(t<=0):
        s = v0 * t
        r = symbols('r')
        fr = r * sqrt(r * r + a * a) / 2 + a * a * ln(
            (r + sqrt(r * r + a * a)) / (r0 + sqrt(r0 * r0 + a * a))) / 2 - r0 * sqrt(
            r0 * r0 + a * a) / 2 + a * s
        r = nsolve(fr, r0 + 2)
        ansr.append(r)
        anssita.append(ansr[0]/a)
        anssite.append(ansr[0]*cos(anssita[0]))
        anssite.append(ansr[0] * sin(anssita[0]))
    elif(t<=(len1+len2)/v0):
        site=location(t)
        x0 = site[0]
        y0 = site[1]
        anssite.append(x0)
        anssite.append(y0)
        ansr.append(sqrt(x0*x0+y0*y0))
        theta = atan(y0/x0)
        if (x0 <= 0):
            theta += pi
        anssita.append(theta)
    else:
        s = v0 * t - len1 - len2
        r = symbols('r')
        fr = r * sqrt(r * r + a * a) / 2 + a * a * ln(
            (r + sqrt(r * r + a * a)) / (r0 + sqrt(r0 * r0 + a * a))) / 2 - r0 * sqrt(
            r0 * r0 + a * a) / 2 - a * s
        r = nsolve(fr, r0 + 2)
        ansr.append(r)
        anssita.append(ansr[0]/a+pi)
        anssite.append(ansr[0]*cos(anssita[0]))
        anssite.append(ansr[0] * sin(anssita[0]))
    x0=anssite[0]
    y0=anssite[1]
    print(ansr[0]-a*anssita[0])
    if((-eps<ansr[0]-a*anssita[0]<eps and ansr[0]>=450) or (ansr[0]<=450 and y0<=-92.2623184486086)):

        r = symbols('r')
        fr = (r * cos(r / a) - x0) *(r * cos(r / a) - x0) + (r * sin(r / a) - y0) *(r * sin(r / a) - y0)- l_head *l_head
        ansr.append(nsolve(fr, max(ansr[0]+1,450)))
        anssita.append(ansr[1]/a)
        ansx = ansr[1] * cos(anssita[1])
        ansy = ansr[1] * sin(anssita[1])
        anssite.append(ansx.evalf())
        anssite.append(ansy.evalf())
        print([ansx.evalf(),ansy.evalf()])
    elif(ansr[0]<=450 and y0<=297.686140859912):
        k=(x1-x0)/(y0-y1)
        b=(x0*x0-x1*x1+y0*y0-y1*y1+r1*r1-l_head*l_head)/(2*(y0-y1))
        tmpa = 1+k*k
        tmpb =2*k*(b-y0) - 2*x0
        tmpc = x0*x0+(b-y0)*(b-y0)-l_head*l_head
        ansx1 = (-tmpb+sqrt(tmpb*tmpb-4*tmpa*tmpc))/(2*tmpa)
        ansy1 = k*ansx1+b
        ansx2 = (-tmpb-sqrt(tmpb*tmpb-4*tmpa*tmpc))/(2*tmpa)
        ansy2 = k*ansx2+b
        theta1 = atan(ansy1/ansx1)
        if(ansx1<=0):
            theta1 += pi
        theta2 = atan(ansy2/ansx2)
        if(ansx2 <= 0):
            theta2 += pi
        if(theta1>anssita[0] and ansy1 > ansx1*tan(r0/a)):
            ansx = ansx1
            ansy = ansy1
        else:
            ansx = ansx2
            ansy = ansy2
        ansx = ansx.evalf()
        anssite.append(ansx)
        ansy = ansy.evalf()
        anssite.append(ansy)
        ansr.append(sqrt(ansx*ansx+ansy*ansy))
        theta = atan(ansy/ansx)
        if(ansx<=0):
            theta += pi
        anssita.append(theta)
    elif(ansr[0]<=467.131999862687):
        k=(x2-x0)/(y0-y2)
        b=(x0*x0-x2*x2+y0*y0-y2*y2+r2*r2-l_head*l_head)/(2*(y0-y2))
        tmpa = 1+k*k
        tmpb =2*k*(b-y0) - 2*x0
        tmpc = x0*x0+(b-y0)*(b-y0)-l_head*l_head
        ansx1 = (-tmpb+sqrt(tmpb*tmpb-4*tmpa*tmpc))/(2*tmpa)
        ansy1 = k*ansx1+b
        ansx2 = (-tmpb-sqrt(tmpb*tmpb-4*tmpa*tmpc))/(2*tmpa)
        ansy2 = k*ansx2+b
        ansr1 = sqrt(ansx1*ansx1+ansy1*ansy1)
        ansr2 = sqrt(ansx2*ansx2+ansy2*ansy2)
        if(ansr1<ansr[0] and ansy1 < ansx1*tan(r0/a)):
            ansx = ansx1
            ansy = ansy1
        else:
            ansx = ansx2
            ansy = ansy2
        
        ansx=ansx.evalf()
        anssite.append(ansx)
        ansy = ansy.evalf()
        anssite.append(ansy)
        ansr.append(sqrt(ansx*ansx+ansy*ansy))
        theta = atan(ansy/ansx)
        if(ansx<=0):
            theta += pi
        anssita.append(theta)
    else:
        r = symbols('r')
        fr = (r * cos(r /a+pi) - x0) *(r * cos(r /a+pi) - x0) + (r * sin(r / a+pi) - y0) *(r * sin(r / a+pi) - y0)- l_head *l_head
        ansr.append(nsolve(fr, ansr[0] - 2))
        anssita.append(ansr[1]/a+pi)
        ansx = ansr[1] * cos(anssita[1])
        ansy = ansr[1] * sin(anssita[1])
        anssite.append(ansx.evalf())
        anssite.append(ansy.evalf())
    for i in range(1, 224):
        x0 = ansr[i] * cos(anssita[i])
        y0 = ansr[i] * sin(anssita[i])
        if((-eps<ansr[i]-a*anssita[i]<eps and ansr[i]>=450)or (ansr[i]<=450 and y0<=-221.624651082794)):
            r = symbols('r')
            fr = (r * cos(r / a) - x0) *(r * cos(r / a) - x0) + (r * sin(r / a) - y0) *(r * sin(r / a) - y0)- l_body *l_body
            ansr.append(nsolve(fr, max(ansr[i]+2,450)))
            anssita.append(ansr[i+1]/a)
            ansx = ansr[i+1] * cos(anssita[i+1])
            ansy = ansr[i+1] * sin(anssita[i+1])
            anssite.append(ansx.evalf())
            anssite.append(ansy.evalf())
        elif(y0<=250 and x0<=255.392637737839 and ansr[i]<=450):
            k=(x1-x0)/(y0-y1)
            b=(x0*x0-x1*x1+y0*y0-y1*y1+r1*r1-l_body*l_body)/(2*(y0-y1))
            tmpa = 1+k*k
            tmpb =2*k*(b-y0) - 2*x0
            tmpc = x0*x0+(b-y0)*(b-y0)-l_body*l_body
            ansx1 = (-tmpb+sqrt(tmpb*tmpb-4*tmpa*tmpc))/(2*tmpa)
            ansy1 = k*ansx1+b
            ansx2 = (-tmpb-sqrt(tmpb*tmpb-4*tmpa*tmpc))/(2*tmpa)
            ansy2 = k*ansx2+b
            theta1 = atan(ansy1/ansx1)
            if(ansx1<=0):
                theta1 += pi
            theta2 = atan(ansy2/ansx2)
            if(ansx2 <= 0):
                theta2 += pi
            if(theta1>anssita[i] and ansy1 > ansx1*tan(r0/a)):
                ansx = ansx1
                ansy = ansy1
            else:
                ansx = ansx2
                ansy = ansy2
            
            ansx=ansx.evalf()
            anssite.append(ansx)
            ansy = ansy.evalf()
            anssite.append(ansy)
            ansr.append(sqrt(ansx*ansx+ansy*ansy))
            theta = atan(ansy/ansx)
            if(ansx<=0):
                theta += pi
            anssita.append(theta)
        elif(ansr[i]<=459.850636532029):
            k=(x2-x0)/(y0-y2)
            b=(x0*x0-x2*x2+y0*y0-y2*y2+r2*r2-l_body*l_body)/(2*(y0-y2))
            tmpa = 1+k*k
            tmpb =2*k*(b-y0) - 2*x0
            tmpc = x0*x0+(b-y0)*(b-y0)-l_body*l_body
            ansx1 = (-tmpb+sqrt(tmpb*tmpb-4*tmpa*tmpc))/(2*tmpa)
            ansy1 = k*ansx1+b
            ansx2 = (-tmpb-sqrt(tmpb*tmpb-4*tmpa*tmpc))/(2*tmpa)
            ansy2 = k*ansx2+b
            ansr1 = sqrt(ansx1*ansx1+ansy1*ansy1)
            ansr2 = sqrt(ansx2*ansx2+ansy2*ansy2)
            if(ansr1<ansr[i] and ansy1 < ansx1*tan(r0/a)):
                ansx = ansx1
                ansy = ansy1
            else:
                ansx = ansx2
                ansy = ansy2
            
            ansx=ansx.evalf()
            anssite.append(ansx)
            ansy = ansy.evalf()
            anssite.append(ansy)
            ansr.append(sqrt(ansx*ansx+ansy*ansy))
            theta = atan(ansy/ansx)
            if(ansx<=0):
                theta += pi
            anssita.append(theta)
        else:
            r = symbols('r')
            fr = (r * cos(r /a+pi) - x0) *(r * cos(r /a+pi) - x0) + (r * sin(r / a+pi) - y0) *(r * sin(r / a+pi) - y0)- l_body *l_body
            ansr.append(nsolve(fr, ansr[i] - 2))
            anssita.append(ansr[i+1]/a+pi)
            ansx = ansr[i+1] * cos(anssita[i+1])
            ansy = ansr[i+1] * sin(anssita[i+1])
            anssite.append(ansx.evalf())
            anssite.append(ansy.evalf())
    return anssite
wb = openpyxl.load_workbook('result4.xlsx')
sheet = wb['位置']
t=10
while(t<=100):
    ans = ans_site(t)
    for i in range(448):
        tmp = ans[i] / 100
        sheet.cell(i + 2, int(t+100+2)).value = str(round(tmp, 6))
    t+=1
wb.save('result4.xlsx')
