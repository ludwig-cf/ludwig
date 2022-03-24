import csv
import numpy as np
import matplotlib.pyplot as plt

NVESICLES = 1
NATOMS = 163

length = 1.0
epsilon = 1e-3

BOND_LENGTH = 2.0

nbonds= 6
nbonds2 = 6
nbonds3 = 3

XSHIFT = 25
YSHIFT = 25
ZSHIFT = 25

# Additional attributes 
indices = np.arange(1,NATOMS+1,1,dtype=int)

nConnec = np.zeros((NATOMS*NVESICLES), dtype = int)
nConnec2 = np.zeros((NATOMS*NVESICLES), dtype = int)
nConnec3 = np.zeros((NATOMS*NVESICLES), dtype = int)
Connec = np.zeros((NATOMS, nbonds), dtype = int)
Connec2 = np.zeros((NATOMS, nbonds2), dtype = int)
Connec3 = np.zeros((NATOMS, nbonds3), dtype = int)
iscentre = np.zeros((NATOMS*NVESICLES), dtype = int)
indexcentre = np.zeros((NATOMS*NVESICLES), dtype = int)
phi_production = np.zeros((NATOMS*NVESICLES))
localmobility = np.zeros((NATOMS*NVESICLES))
localrange = np.zeros((NATOMS*NVESICLES))

localrange[::] = 1.1*BOND_LENGTH
localrange[-1] = 1.1*BOND_LENGTH

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
xyz = file_to_array("trisphere.xyz")

for p0 in xyz.T[:20]:
  for i, p1 in enumerate(xyz.T):
    dr = dist(p0, p1)
    if dr < 0.4: print(i, dr)


# Connectivities
for i in range(NATOMS):
  bondmade = 0
  bondmade2 = 0
  for j in range(NATOMS):
    if i == j: continue

    dr = dist(xyz.T[i], xyz.T[j])

    if dr < icosphere_small_l + epsilon and bondmade < nbonds:
      Connec[i][bondmade] = np.int_(j)
      nConnec[i] += 1
      bondmade += 1

    if icosphere_large_l - epsilon < dr < icosphere_large_l + epsilon and bondmade < nbonds2:
      Connec2[i][bondmade2] = np.int_(j)
      nConnec2[i] += 1
      bondmade2 += 1 

for i in range(len(Connec)):
  for j in range(len(Connec[i])):
    if Connec[i][j] == 0: continue
    Connec[i][j] += 1
for i in range(len(Connec2)):
  for j in range(len(Connec2[i])):
    if Connec2[i][j] == 0: continue
    Connec2[i][j] += 1
for i in range(len(Connec3)):
  for j in range(len(Connec3[i])):
    if Connec3[i][j] == 0: continue
    Connec3[i][j] += 1

Connec = np.array(Connec)
Connec2 = np.array(Connec2)
Connec3 = np.array(Connec3)



# Renormalize distances so that the smallest of the two harmonic bonds has l_0=1
factor = BOND_LENGTH / icosphere_small_l
xyz = rescale(xyz, factor)

# For each particle, print adjacent lengths and angle
angles = []
for i, p0 in enumerate(xyz.T):

  #neighbours distance
  if (Connec[i][0] == 0): p1 = p0
  else: p1 = xyz.T[Connec[i][0] - 1]

  if (Connec[i][1] == 0): p2 = p0
  else: p2 = xyz.T[Connec[i][1] - 1]

  if (Connec[i][2] == 0): p3 = p0
  else: p3 = xyz.T[Connec[i][2] - 1]

  #neighbours angle
  cosine1 = np.dot(p1-p0, p2-p0)/(dist(p0,p1)*dist(p0,p2)) #angle 012
  cosine2 = np.dot(p2-p0, p3-p0)/(dist(p0,p2)*dist(p0,p3)) #angle 023
  cosine3 = np.dot(p1-p0, p3-p0)/(dist(p0,p1)*dist(p0,p3)) #angle 013
   
  print(i,"012", np.arccos(cosine1))
  print(i,"023", np.arccos(cosine2))
  print(i,"013", np.arccos(cosine3))
  angles.append(np.arccos(cosine1))
  angles.append(np.arccos(cosine2))
  angles.append(np.arccos(cosine3))

angles2 = []
# For each particle, print adjacent lengths and angle
for i, p0 in enumerate(xyz.T):

  #neighbours distance
  if (Connec2[i][0] == 0): p1 = p0
  else: p1 = xyz.T[Connec2[i][0] - 1]

  if (Connec2[i][1] == 0): p2 = p0
  else: p2 = xyz.T[Connec2[i][1] - 1]

  if (Connec2[i][2] == 0): p3 = p0
  else: p3 = xyz.T[Connec2[i][2] - 1]

  #neighbours angle
  cosine1 = np.dot(p1-p0, p2-p0)/(dist(p0,p1)*dist(p0,p2)) #angle 012
  cosine2 = np.dot(p2-p0, p3-p0)/(dist(p0,p2)*dist(p0,p3)) #angle 023
  cosine3 = np.dot(p1-p0, p3-p0)/(dist(p0,p1)*dist(p0,p3)) #angle 013
  
  print(i,"012", np.arccos(cosine1))
  print(i,"023", np.arccos(cosine2))
  print(i,"013", np.arccos(cosine3))

  angles2.append(np.arccos(cosine1))
  angles2.append(np.arccos(cosine2))
  angles2.append(np.arccos(cosine3))

plt.hist(angles)
plt.hist(angles2)
plt.show()

#Other attributes
iscentre[0] = 1 #0, NATOMS, etc...
indexcentre[0:NATOMS + 1] = 1 #(0, NATOMS-1), (NATOMS, 2*NATOMS), etc..
phi_production[0] = 0.01
localmobility[::] = 0.0
localmobility[-1] = 0.0

xyz[0, :] += XSHIFT
xyz[1, :] += YSHIFT
xyz[2, :] += ZSHIFT

table = np.column_stack((indices, xyz.T, nConnec, Connec, nConnec2, Connec2, nConnec3, Connec3, iscentre.T, indexcentre.T, phi_production.T, localrange.T, localmobility.T))
np.savetxt("latticeHexasphere.txt", table, fmt = '%3d     %3f %3f %3f      %3d %3d %3d %3d     %3d %3d %3d %3d     %3d %3d %3d %3d     %3d %3d      %3f     %3f %3f ')
