import csv
import numpy as np
import matplotlib.pyplot as plt

plot = 0

NVESICLES = 1
NATOMS = 643

sphere_l = 0.17

# or choose vesicle radius
RADIUS = 10.0

#print(str(BOND_LENGTH) + "\n" + str(BOND_LENGTH*sphere_l) + "\n" + str(BOND_LENGTH*RADIUS0))

nbonds= 7

XSHIFT = 25
YSHIFT = 25
ZSHIFT = 25

#M27 = np.array([0.061374, -0.467955, 0.843134])
M27 = np.array([-0.218611, 0.822090, 0.458258])
M = np.array([1, 0, 0]) # Vesicle oriented towards X (hole towards -X)

M = M / np.sqrt(np.sum(M**2))
M27 = M27 / np.sqrt(np.sum(M27**2))
 
v = np.cross(M, M27)
s = np.sqrt(np.sum(v**2))
c = np.dot(M27, M)

matvx = np.zeros((3,3))
matvx[0][1] = -v[2]
matvx[0][2] = v[1]
matvx[1][0] = v[2]
matvx[1][2] = -v[0]
matvx[2][0] = -v[1]
matvx[2][1] = v[0]
matvx2 = np.matmul(matvx, matvx)
R = np.eye(3) + matvx + matvx2 * (1/(1+c)) 


# Additional attributes 
indices = np.arange(1,NATOMS+1,1,dtype=int)

nConnec = np.zeros((NATOMS*NVESICLES), dtype = int)
Connec = np.zeros((NATOMS, nbonds), dtype = int)
Connecdist = np.zeros((NATOMS, nbonds), dtype = float)

iscentre = np.zeros((NATOMS*NVESICLES), dtype = int)
ishole = np.zeros((NATOMS*NVESICLES), dtype = int)
indexcentre = np.ones((NATOMS*NVESICLES), dtype = int)

def file_to_array(filename):
  x, y, z = [], [], []

  f = open(filename, "r")
  for line in f.readlines()[:]:
    x.append(line.split()[0])
    y.append(line.split()[1])
    z.append(line.split()[2])

    x = list(map(float, x))
    y = list(map(float, y))
    z = list(map(float, z))
  coords = np.concatenate(([x],[y],[z]))
  return(np.array(coords))

# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def rescale(arr, factor):
  arr = np.array(arr) 
  shape = np.shape(arr)
  crepe = np.array([factor * arr.flatten()[i] for i in range(len(arr.flatten())) ] )
  return crepe.reshape(shape)
  
def dist(x,y):
  dr = np.sqrt(np.sum((x-y)**2))
  return dr

# Coordinates
xyz = file_to_array("4sphere.xyz")

#Renormalize distances so that the smallest of the two harmonic bonds has l_0=1
xyz = rescale(xyz, RADIUS)

# Connectivities
for i in range(NATOMS):
  bondmade = 0
  for j in range(NATOMS):
    dr = dist(xyz.T[i], xyz.T[j])
    if i == j: continue

    elif dr < sphere_l*RADIUS:
      Connec[i][bondmade] = np.int_(j) + 1
      Connecdist[i][bondmade] = dr

      nConnec[i] += 1
      bondmade += 1

  for j in range(NATOMS):
    if j == 0:
      dr = dist(xyz.T[i], xyz.T[j])
      Connec[i][bondmade] = np.int_(j) + 1
      Connecdist[i][bondmade] = dr

      nConnec[i] += 1
      bondmade += 1



Connec = np.array(Connec)

#Renormalize distances so that the smallest of the two harmonic bonds has l_0=1
#xyz = rescale(xyz, RADIUS)

#Other attributes
iscentre[0] = 1 #0, NATOMS, etc...
ishole[NATOMS - 1] = 1 #0, NATOMS, etc...

xyzt = xyz.T
for i, vec in enumerate(xyzt):
  newvec = np.dot(R.T, vec)
  xyz[0][i] = newvec[0]
  xyz[1][i] = newvec[1]
  xyz[2][i] = newvec[2]

xyz[0, :] += XSHIFT
xyz[1, :] += YSHIFT
xyz[2, :] += ZSHIFT

print(indices.shape, xyz.T.shape, nConnec.shape, Connec.shape, iscentre.T.shape, ishole.T.shape, indexcentre.T.shape)

table = np.column_stack((indices, xyz.T, nConnec, Connec, Connecdist, iscentre.T, ishole.T, indexcentre.T))
print(indices[:5], Connec[:5])
np.savetxt("lattice4sphere.txt", table, fmt = '%3d     %3f %3f %3f      %3d %3d %3d %3d %3d %3d %3d %3d   %3f %3f %3f %3f %3f %3f %3f    %3d %3d %3d')

if plot:
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
