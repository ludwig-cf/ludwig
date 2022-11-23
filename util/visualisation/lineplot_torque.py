import numpy as np 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from textwrap import wrap

DATA_FOLDER="graph_data"

with open(DATA_FOLDER+"/"+'txOverTimeAndFolders') as f:
  lines = f.readlines()
  header = lines[0]

txs = np.loadtxt(DATA_FOLDER+"/"+"txOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
tys = np.loadtxt(DATA_FOLDER+"/"+"tyOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
tzs = np.loadtxt(DATA_FOLDER+"/"+"tzOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)

t = np.linspace(1, 100, 20)
labels = header.split(",")

plt.figure(figsize = (15,10))
for i, (tx, label) in enumerate(zip(txs, labels)):

  plt.plot(t, tx, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$t_x$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 16)
plt.tight_layout()
savename="tx.png"
plt.savefig(savename)
plt.clf()

for i, (ty, label) in enumerate(zip(tys, labels)):

  plt.plot(t, ty, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$t_y$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 16)
plt.tight_layout()
savename="ty.png"
plt.savefig(savename)
plt.clf()

for i, (tz, label) in enumerate(zip(tzs, labels)):

  plt.plot(t, tz, '--o', label=label, color=cm.Blues((i+1)/len(labels)))
  plt.xlabel(r"\emph{timestep x1000}", fontsize = 16)
  plt.ylabel(r"\emph{$t_z$ (l.u.)", fontsize = 16)

plt.legend(fontsize = 16)
plt.tight_layout()
savename="tz.png"
plt.savefig(savename)
plt.clf()
