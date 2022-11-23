import numpy as np 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap
DATA_FOLDER="graph_data"

with open(DATA_FOLDER+"/"+'vxOverTimeAndFolders') as f:
  lines = f.readlines()
  header = lines[0]

vxs = np.loadtxt(DATA_FOLDER+"/"+"vxOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
vys = np.loadtxt(DATA_FOLDER+"/"+"vyOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
vzs = np.loadtxt(DATA_FOLDER+"/"+"vzOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)

labels = header.split(",")

t = np.linspace(1, 100, 20)

plt.figure(figsize = (15,10))

for i, (vx, label) in enumerate(zip(vxs, labels)):

  plt.plot(t, vx, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$m_x$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 12)
plt.tight_layout()
savename="vx.png"
plt.savefig(savename)
plt.clf()


for i, (vy, label) in enumerate(zip(vys, labels)):

  plt.plot(t, vy, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$m_y$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 12)
plt.tight_layout()
savename="vy.png"
plt.savefig(savename)
plt.clf()


for i, (vz, label) in enumerate(zip(vzs, labels)):

  plt.plot(t, vz, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$m_z$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 12)
plt.tight_layout()
savename="vz.png"
plt.savefig(savename)
plt.clf()

