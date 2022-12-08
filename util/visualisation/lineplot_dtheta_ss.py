import numpy as np 
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from basic_units import radians, degrees, cos
from textwrap import wrap
DATA_FOLDER="graph_data"

with open(DATA_FOLDER+"/"+'xcOverTimeAndFolders') as f:
  lines = f.readlines()
  header = lines[0]

labels = header.split(",")

tangential_forces = np.array([1e-3, 1e-4, 1e-5, 1e-6])

xc = np.loadtxt(DATA_FOLDER+"/"+"xcOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
yc = np.loadtxt(DATA_FOLDER+"/"+"ycOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
zc = np.loadtxt(DATA_FOLDER+"/"+"zcOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)

xh = np.loadtxt(DATA_FOLDER+"/"+"xhOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
yh = np.loadtxt(DATA_FOLDER+"/"+"yhOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)
zh = np.loadtxt(DATA_FOLDER+"/"+"zhOverTimeAndFolders", delimiter=',', unpack=True, skiprows = 1)

# Transformation to spherical coordinates with origin = center of the vesicle
# Fields are ordered in: [ Folders, Time ]
# (xc, yc, zc) is the bead centre of the vesicle in global coordinates
# (xh, yh, zh) is the bead hole of the vesicle in global coordinates
# transformation to the frame of reference of the vesicle (xh-xc, yh-yc, zh-zc) => (r, theta, phi) 

rs = np.sqrt((xh-xc)**2+(yh-yc)**2+(zh-zc)**2)

thetas = np.zeros(xc.shape)
phis = np.zeros(xc.shape)

non_zero = xh-xc != 0.0
thetas[non_zero] = np.arctan2(yh[non_zero]-yc[non_zero], xh[non_zero]-xc[non_zero])
thetas[~non_zero] = np.pi/2

non_zero = zh-zc != 0.0
phis[non_zero] = np.arctan2(np.sqrt((xh[non_zero]-xc[non_zero])**2+(yh[non_zero]-yc[non_zero])**2),zh[non_zero]-zc[non_zero])
phis[~non_zero] = np.pi/2

# angular velocity
# last derivative is incorrect with np roll boundary conditions so we fix it
dthetas = np.roll(thetas, (0, -1)) - thetas; dthetas[:, -1] = dthetas[:, -2]
dphis = np.roll(phis, (0, -1)) - phis; dphis[:, -1] = dphis[:, -2]

plt.figure(figsize=(10, 6))
plt.loglog(tangential_forces, -dthetas[0:4,-1], '--', color='black')
plt.ylabel(r"\emph{$- \log(\dot \theta)$ (radian/s)}", fontsize = 16)
plt.xlabel(r"\emph{tangential force magnitude}", fontsize = 16)

plt.legend(fontsize = 12)
savename="dtheta_ss_fullerene.png"
plt.savefig(savename)
plt.clf()


plt.figure(figsize=(10, 6))
plt.loglog(tangential_forces, -dthetas[4:8,-1], '--', color='black')
plt.ylabel(r"\emph{$- \log(\dot \theta)$ (radian/s)}", fontsize = 16)
plt.xlabel(r"\emph{tangential force magnitude}", fontsize = 16)

plt.legend(fontsize = 12)
savename="dtheta_ss_hexasphere.png"
plt.savefig(savename)
plt.clf()


plt.figure(figsize=(10, 6))
plt.loglog(tangential_forces, -dthetas[8:12,-1], '--', color='black')
plt.ylabel(r"\emph{$- \log(\dot \theta)$ (radian/s)}", fontsize = 16)
plt.xlabel(r"\emph{tangential force magnitude}", fontsize = 16)

plt.legend(fontsize = 12)
savename="dtheta_ss_trisphere.png"
plt.savefig(savename)
plt.clf()

