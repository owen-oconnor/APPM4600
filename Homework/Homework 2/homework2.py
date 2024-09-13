import numpy as np
import matplotlib.pyplot as plt

#4a

t = np.arange(0, np.pi, np.pi/30)
y = np.cos(t)

sum = 0
for i in range(1, len(t)):
    sum += t[i]*y[i]
    print(t[i], y[i], sum)

print(f'the sum is {sum}')

#4b


