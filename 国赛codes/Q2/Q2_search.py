from sympy import *
eps = 1e-12
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
def bin_search(left,right):
    mid = (left+right)/2
    ans=ans_site(mid)
    cosx0=(ans[2]-ans[0])/2.86
    cosy0=(ans[3]-ans[1])/2.86
    x1=ans[0]-0.275*cosx0
    y1=ans[1]-0.275*cosy0
    x0=x1+0.15*cosy0
    y0=y1-0.15*cosx0
    dis = distance(x0,y0,ans[16],ans[17],ans[18],ans[19])
    if -eps<dis-15<eps:
        return mid
    elif dis-15<0:
        return bin_search(left,mid)
    else:
        return bin_search(mid,right)
def distance(x0,y0,x1,y1,x2,y2):
    dis = abs(x0*(y2-y1)+y0*(x1-x2)+(x2*y1-x1*y2))/sqrt((y1-y2)*(y1-y2)+(x1-x2)*(x1-x2))
    return dis

print(bin_search(412,414))


