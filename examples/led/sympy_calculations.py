from sympy import symbols
from sympy import sin, cos, pi, diff, simplify

# function in x direction: wp.sin(4.*wp.pi*(x-0.3*t)) * wp.cos(2.*wp.pi*(y-0.8*t)) * wp.sin(4.*wp.pi*(t-0.1))
# function in x direction: wp.cos(4.*wp.pi*(x-0.7*t)) * wp.sin(2.*wp.pi*(y-0.1*t)) * wp.cos(4.*wp.pi*(t+0.4))
x, y, t = symbols('x y t')

ux = sin(4*pi*(x-0.3*t)) * cos(2*pi*(y-0.8*t)) * sin(4*pi*(t-0.1))
uy = cos(4*pi*(x-0.7*t)) * sin(2*pi*(y-0.1*t)) * cos(4*pi*(t+0.4))

ux_t = diff(ux, t)
uy_t = diff(uy, t)

print(ux_t)
print(uy_t)
