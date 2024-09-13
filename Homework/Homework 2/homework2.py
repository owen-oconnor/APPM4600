import numpy as np
import matplotlib.pyplot as plt
import random

#4a

t = np.arange(0, np.pi, np.pi/30)
y = np.cos(t)

sum = 0
for i in range(1, len(t)):
    sum += t[i]*y[i]

print(f'the sum is {sum}')

#4b
R = 1.2
dr = 0.1
f = 15
p = 0
theta = np.linspace(0, 2*np.pi, 100)
x = R*(1+dr*(np.sin(f*theta + p)))*np.cos(theta)
y = R*(1+dr*(np.sin(f*theta + p)))*np.sin(theta)


plt.plot(x, y)
plt.show()

dr = 0.05
for i in range(10):
    R = i
    f = 2 + i
    p = random.uniform(0,2)
    x = R*(1+dr*(np.sin(f*theta + p)))*np.cos(theta)
    y = R*(1+dr*(np.sin(f*theta + p)))*np.sin(theta)
    plt.plot(x, y)

plt.show()


