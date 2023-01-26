import csv
import numpy as np
import matplotlib.pyplot as plt
import utils

NVESICLES = 1
NATOMS = 643

sphere_l = 0.17

# or choose vesicle radius
RADIUS = 4.0

#print(str(BOND_LENGTH) + "\n" + str(BOND_LENGTH*sphere_l) + "\n" + str(BOND_LENGTH*RADIUS0))

nbonds= 7

XSHIFT = 8.0
YSHIFT = 8.0
ZSHIFT = 8.0

mx=-1
my=0
mz=0

#nx=XXXnxXXX
#ny=XXXnyXXX
#nz=XXXnzXXX

M = np.array([mx, my, mz]) # Vesicle oriented towards X (hole towards -X)
M = M / np.sqrt(np.sum(M**2))

#N = np.array([nx, ny, nz]) # Vesicle oriented towards X (hole towards -X)
#N = N / np.sqrt(np.sum(N**2))

# First orient M164 towards N then M643 towards M
#R164 = utils.rotate(COL164, N)

# Additional attributes 
indices = np.arange(1,NATOMS+1,1,dtype=int)

nConnec = np.zeros((NATOMS*NVESICLES), dtype = int)
Connec = np.zeros((NATOMS, nbonds), dtype = int)
Connecdist = np.zeros((NATOMS, nbonds), dtype = float)

iscentre = np.zeros((NATOMS*NVESICLES), dtype = int)
ishole = np.zeros((NATOMS*NVESICLES), dtype = int)
indexcentre = np.ones((NATOMS*NVESICLES), dtype = int)


# Coordinates
xyz = utils.file_to_array("rawfiles/trisphere.xyz")

#Renormalize distances so that the smallest of the two harmonic bonds has l_0=1
#xyz = utils.rescale(xyz, RADIUS)


# Connectivities
for i in range(NATOMS):
  bondmade = 0
  for j in range(NATOMS):
    dr = utils.dist(xyz.T[i], xyz.T[j])
    if i == j: continue

    elif dr < sphere_l:
      Connec[i][bondmade] = np.int_(j) + 1
      Connecdist[i][bondmade] = dr*RADIUS

      nConnec[i] += 1
      bondmade += 1

  for j in range(NATOMS):
    if j == 0:
      dr = utils.dist(xyz.T[i], xyz.T[j])
      Connec[i][bondmade] = np.int_(j) + 1
      Connecdist[i][bondmade] = dr*RADIUS

      nConnec[i] += 1
      bondmade += 1

Connec = np.array(Connec)

#Renormalize distances so that the smallest of the two harmonic bonds has l_0=1


#Other attributes
iscentre[0] = 1 #0, NATOMS, etc...
ishole[NATOMS - 1] = 1 #0, NATOMS, etc...

xyzt = xyz.T
#for i, vec in enumerate(xyzt):
#  newvec = np.dot(R164.T, vec)
#  xyz[0][i] = newvec[0]
#  xyz[1][i] = newvec[1]
#  xyz[2][i] = newvec[2]

COL643 = xyz[:,642]
COL643 /= np.sqrt(np.sum(COL643**2))

R643 = utils.rotate(COL643, M)

for i, vec in enumerate(xyzt):
  newvec = np.dot(R643.T, vec)
  xyz[0][i] = newvec[0]
  xyz[1][i] = newvec[1]
  xyz[2][i] = newvec[2]

xyz = utils.rescale(xyz, RADIUS)

xyz[0, :] += XSHIFT
xyz[1, :] += YSHIFT
xyz[2, :] += ZSHIFT

table = np.column_stack((indices, xyz.T, nConnec, Connec, Connecdist, iscentre.T, ishole.T, indexcentre.T))
np.savetxt("latticeTrisphere.txt", table, fmt = '%3d     %3f %3f %3f      %3d %3d %3d %3d %3d %3d %3d %3d   %3f %3f %3f %3f %3f %3f %3f    %3d %3d %3d')



if 0:
  dists = []
  lendists = np.zeros((NATOMS))
  for i in range(xyz.shape[1]):
    coordi = xyz[:,i]
    for j,coordj in enumerate(xyz.T):
      if (i == j): continue
      else:
        dist = np.sqrt(np.sum((coordi - coordj)**2 ))
        if (dist < 0.17): 
          lendists[i] += 1
          dists.append(dist) 

#print(lendists)


