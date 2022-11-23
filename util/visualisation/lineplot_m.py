import numpy as np 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap
DATA_FOLDER="graph_data"

with open(DATA_FOLDER+"/"+'mxOverTimeAndFolders') as f:
  lines = f.readlines()
  header = lines[0]

mxs = np.loadtxt(DATA_FOLDER+"/"+"mxOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
mys = np.loadtxt(DATA_FOLDER+"/"+"myOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
mzs = np.loadtxt(DATA_FOLDER+"/"+"mzOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)

labels = header.split(",")

t = np.linspace(1, 100, 20)

plt.figure(figsize = (15,10))

for i, (mx, label) in enumerate(zip(mxs, labels)):

  plt.plot(t, mx, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$m_x$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 12)
plt.tight_layout()
savename="mx.png"
plt.savefig(savename)
plt.clf()


for i, (my, label) in enumerate(zip(mys, labels)):

  plt.plot(t, my, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$m_y$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 12)
plt.tight_layout()
savename="my.png"
plt.savefig(savename)
plt.clf()


for i, (mz, label) in enumerate(zip(mzs, labels)):

  plt.plot(t, mz, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$m_z$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 12)
plt.tight_layout()
savename="mz.png"
plt.savefig(savename)
plt.clf()

