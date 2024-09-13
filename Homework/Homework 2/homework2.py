import numpy as np
import matplotlib.pyplot as plt
import random
import math

#3c
def f(x):
    y = math.e**x
    return y - 1

def f2(x): # 2nd order taylor approx of e^x - 1
    y = x + 0.5*(x**2)
    return y

print(f(9.999999500000e-10)) # loses precision
print(f2(9.999999500000e-10)) # preserves full precision of digit values 

################################

#4a

t = np.arange(0, np.pi, np.pi/30)
y = np.cos(t)

sum = 0
for i in range(1, len(t)):
    sum += t[i]*y[i]

print(f'the sum is {sum}')

##########################################

#4b
R = 1.2
dr = 0.1
f = 15
p = 0
theta = np.linspace(0, 2*np.pi, 100)
x = R*(1+dr*(np.sin(f*theta + p)))*np.cos(theta)
y = R*(1+dr*(np.sin(f*theta + p)))*np.sin(theta)


plt.plot(x, y) # first parametric plot
plt.show()

dr = 0.05
for i in range(1, 11): # ensure 10 plots as if index starts at 0, R=0 plot will be empty
    R = i
    f = 2 + i
    p = random.uniform(0,2)
    x = R*(1+dr*(np.sin(f*theta + p)))*np.cos(theta)
    y = R*(1+dr*(np.sin(f*theta + p)))*np.sin(theta)
    plt.plot(x, y)

plt.show()


