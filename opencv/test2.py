import numpy as np
import matplotlib.pyplot as plt

x = range(10)
y = range(10)

fig, ax = plt.subplots(2)

for sp in ax:
    sp.plot(x, y)

plt.show()