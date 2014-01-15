#!/usr/bin/python
""" Given oelb lattice file, plot potential
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import getopt
import urllib
import re
import math

 
def Usage():
    print 'Usage: ',sys.argv[1],  '-f oelb_latt.dat -L L_box'
    sys.exit()

optlist, args = getopt.getopt(sys.argv[1:],'f:L:')
dict_opts     = dict( optlist )

try:
    file     = dict_opts['-f'] 
    L        = float(dict_opts['-L']) 
except:
    Usage()

f = open(file, 'r')

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# WRONG WAY TO INITIALIZE MATRIX AS LISTS ARE FAKE (I.E. SAME LIST POINTED BY ALLO ROWS)
#Z = list( 0 for i in np.arange(0, L) )
#Z = list( Z for i in np.arange(0, L) )
# RIGHT WAY TO INITIALIZE MATRIX
Z = []
for i in np.arange(0, L): Z.append( [0.0] * int(L)  )
for line in f.readlines():
    fields = line.split()
    if ( int(fields[2]) == L / 2.0 ):
        xs = int(fields[0])
        ys = int(fields[1])
        zs = float(fields[7])
        Z[ys-1][xs-1] = zs

c = 'r'
X = np.arange( 0, L )
Y = np.arange( 0, L )
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, np.array(Z), rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlim3d(0, L)
ax.set_ylim3d(0, L)
z_lim = max(max(Z))
ax.set_zlim3d(-z_lim - z_lim / L, z_lim + z_lim / L)
plt.show()
fname = '%s.png'%file
fig.savefig(fname)


