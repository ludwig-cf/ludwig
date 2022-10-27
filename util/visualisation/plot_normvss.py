import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap

y = np.loadtxt("vxssOverFolders", delimiter=',', unpack=True)

labels = ["1e-4", "1e-5", "1e-6"]
gradmuext = [1e-4, 1e-5, 1e-6]

for i, (yi, label) in enumerate(zip(y, labels)):
  plt.loglog(gradmuext[i], yi, '--o', label=labels[i], color = 'black')
  plt.title("\n".join(wrap(r"$v_{x,ss}$"+r" for different values of " +r"$\nabla_x\mu_{ext}$", 60)), fontsize = 14)

  plt.xlabel(r"$\nabla_x\mu_{ext}$", fontsize = 12)
  plt.xticks([1e-4,1e-5,1e-6])
  plt.ylabel(r"\emph{speed (l.u.)}", fontsize = 12)

plt.tight_layout()
#plt.show()
plt.savefig("gradmuext_ss.png")
