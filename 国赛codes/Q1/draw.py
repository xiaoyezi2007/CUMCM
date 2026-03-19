from sympy import *

a = 27.5 / pi
r0 = 16 * 55
v0 = 100
r, t = symbols('r t')
plot_implicit(Eq(r * sqrt(r * r + a * a) / 2 + a * a * ln(
    (r + sqrt(r * r + a * a)) / (r0 + sqrt(r0 * r0 + a * a))) / 2 - r0 * sqrt(r0 * r0 + a * a) / 2 + a * v0 * t, 0),
              (t, 0, 300), (r, 400, 880))
