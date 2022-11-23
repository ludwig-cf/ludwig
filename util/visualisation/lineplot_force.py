import numpy as np 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap


with open('fxOverTimeAndFolders') as f:
  lines = f.readlines()
  header = lines[0]


fxs = np.loadtxt("fxOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
fys = np.loadtxt("fyOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
fzs = np.loadtxt("fzOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)

t = np.linspace(1, 100, 100)
labels = header.split(",")

for i, (fx, label) in enumerate(zip(fxs, labels)):

  plt.plot(t, fx, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$f_x$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 16)
plt.tight_layout()
savename="fx.png"
plt.savefig(savename)
plt.clf()

for i, (fy, label) in enumerate(zip(fys, labels)):
  plt.plot(t, fy, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$f_y$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 16)
plt.tight_layout()
savename="fy.png"
plt.savefig(savename)
plt.clf()

for i, (fz, label) in enumerate(zip(fzs, labels)):

  plt.plot(t, fz, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$f_z$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 16)
plt.tight_layout()
savename="fz.png"
plt.savefig(savename)
plt.clf()
