import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap

vxs = np.loadtxt("vxOverTimeAndFolders", delimiter=',', unpack=True)

labels = ["1e-4", "1e-5", "1e-6"]
gradmuext = [1e-4, 1e-5, 1e-6]

for i, (vx, label) in enumerate(zip(vxs, labels)):
  plt.loglog(gradmuext[i], vx[:, -1], '--o', label=labels[i], color = 'black')
  #plt.title("\n".join(wrap(r"$v_{x,ss}$"+r" for different values of " +r"$\nabla_x\mu_{ext}$", 60)), fontsize = 14)

  plt.xlabel(r"$\nabla_x\mu_{ext}$", fontsize = 16)
  plt.xticks([1e-4,1e-5,1e-6])
  plt.ylabel(r"\emph{$v_x$ steady-state (l.u.)}", fontsize = 16)

plt.tight_layout()
#plt.show()
plt.savefig("vxss_gradmu.png")
