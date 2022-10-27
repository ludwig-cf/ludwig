import numpy as np 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap

y = np.loadtxt("vxOverTimeAndFolders", delimiter=',', unpack=True)

t = np.linspace(1, 20, 20)
labels = ["1e-4", "1e-5", "1e-6"]

for i, (yi, label) in enumerate(zip(y, labels)):

  plt.plot(t, yi, '--o', label=labels[i], color=cm.Blues((i+1)/len(labels)))
  plt.axhline(y=0.0, color='black', linestyle='-', alpha=0.1)

  plt.title("\n".join(wrap(r"$v_{x}(t)$"+r" for different values of " +r"$\nabla\mu_{ext}$", 60)), fontsize = 14)
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 12)
  plt.ylabel(r"\emph{speed (l.u.)", fontsize = 12)
  plt.xticks([1,10,20])

plt.legend(fontsize = 14)
plt.tight_layout()
#plt.show()
plt.savefig("gradmu.png")
