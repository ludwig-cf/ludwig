import csv
import numpy as np
import matplotlib.pyplot as plt
import utils


xyz = utils.file_to_array("rawfiles/icosphere.xyz")

if 1:
  dists = []
  coord0 = xyz[:,0]
  for coord in xyz.T[1::]:
    dist = np.sqrt(np.sum((coord0 - coord)**2 ))
    dists.append(dist)

  mean = np.mean(dists)

# This one does not need rescaling
# xyz = utils.rescale(xyz, 1./mean)

if 1:
  dists = []
  coord1 = xyz[:,1]
  for coord in xyz.T:
    dist = np.sqrt(np.sum((coord1 - coord)**2 ))
    dists.append(dist)

  mean = np.mean(dists)
#print(np.array(dists)[np.array(dists) < 0.6])

NVESICLES = 1
NATOMS = 43

RADIUS = 4.0

sphere_l = 0.6

nbonds= 7

XSHIFT = 8
YSHIFT = 8
ZSHIFT = 8

mx = -1
my = 0
mz = 0

M = np.array([mx, my, mz]) # Vesicle oriented towards X (hole towards -X)
M = M / np.sqrt(np.sum(M**2))

# Renormalize the datapoints
if 1:
  dists = []
  lendists = np.zeros((NATOMS))
  coord0 = xyz[:,0]
  for j,coordj in enumerate(xyz.T):
    dist = np.sqrt(np.sum((coord0 - coordj)**2 ))
    if dist > 0.5:
      dists.append(dist)

rawfile_radius = np.mean(dists)
xyz = utils.rescale(xyz, 1./rawfile_radius)



# Additional attributes 
indices = np.arange(1,NATOMS+1,1,dtype=int)

nConnec = np.zeros((NATOMS*NVESICLES), dtype = int)
Connec = np.zeros((NATOMS, nbonds), dtype = int)
Connecdist = np.zeros((NATOMS, nbonds), dtype = float)

iscentre = np.zeros((NATOMS*NVESICLES), dtype = int)
ishole = np.zeros((NATOMS*NVESICLES), dtype = int)
indexcentre = np.ones((NATOMS*NVESICLES), dtype = int)

# Connectivities
for i in range(NATOMS):
  bondmade = 0
  for j in range(NATOMS):
    dr = utils.dist(xyz.T[i], xyz.T[j])
    if i == j: continue

    elif dr < sphere_l:
      Connec[i][bondmade] = np.int_(j) + 1

      nConnec[i] += 1
      bondmade += 1

  for j in range(NATOMS):
    if j == 0:
      dr = utils.dist(xyz.T[i], xyz.T[j])
      Connec[i][bondmade] = np.int_(j) + 1

      nConnec[i] += 1
      bondmade += 1

Connec = np.array(Connec)

#Other attributes
iscentre[0] = 1 #0, NATOMS, etc...
ishole[NATOMS - 1] = 1 #0, NATOMS, etc...

xyzt = xyz.T
#for i, vec in enumerate(xyzt):
#  newvec = np.dot(R164.T, vec)
#  xyz[0][i] = newvec[0]
#  xyz[1][i] = newvec[1]
#  xyz[2][i] = newvec[2]

COL43 = xyz[:,42]
COL43 /= np.sqrt(np.sum(COL43**2))

R43 = utils.rotate(COL43, M)

for i, vec in enumerate(xyzt):
  newvec = np.dot(R43.T, vec)
  xyz[0][i] = newvec[0]
  xyz[1][i] = newvec[1]
  xyz[2][i] = newvec[2]

xyz = utils.rescale(xyz, RADIUS)

# Distances
for i in range(NATOMS):
  bondmade = 0
  for j in range(NATOMS):
    dr = utils.dist(xyz.T[i], xyz.T[j])
    if i == j: continue

    if Connec[i][bondmade] == np.int_(j) + 1:
      Connecdist[i][bondmade] = dr
      bondmade += 1

  for j in range(NATOMS):
    if j == 0:
      dr = utils.dist(xyz.T[i], xyz.T[j])
      if Connec[i][bondmade] == np.int_(j) + 1:
        Connecdist[i][bondmade] = dr
        bondmade += 1

xyz[0, :] += XSHIFT
xyz[1, :] += YSHIFT
xyz[2, :] += ZSHIFT


table = np.column_stack((indices, xyz.T, nConnec, Connec, Connecdist, iscentre.T, ishole.T, indexcentre.T))
np.savetxt("latticeIcosphere.txt", table, fmt = '%3d     %3f %3f %3f      %3d %3d %3d %3d %3d %3d %3d %3d   %3f %3f %3f %3f %3f %3f %3f    %3d %3d %3d')
