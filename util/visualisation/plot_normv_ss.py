import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

y = np.loadtxt("normvss")

mobilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
labels = mobilities

plt.plot(mobilities, y)

plt.title("Steady-state velocity as a function of mobility")
plt.legend()
plt.show()
