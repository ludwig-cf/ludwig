import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

y = np.loadtxt("file", delimiter='	', unpack=True)

t = np.linspace(10000, 500000, 50)
#mobilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
mobilities= [1.0 , 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]#
labels = mobilities


for i, (yi, label) in enumerate(zip(y, labels)):
  plt.plot(t, yi, label=label, color=cm.gray(i / len(labels)))

plt.legend()
plt.show()
