import csv
import numpy as np


NVESICLES = 1
NATOMS = 61

BOND_LENGTH = 4.0
BOND_LENGTH2 = 2.535714*BOND_LENGTH # fullerene radius

NBONDS = 3

NBONDS2 = 6
LINKS2 = [11, 21, 31, 41, 51, 61]

XSHIFT = [30, 48]
YSHIFT = [30, 48]
ZSHIFT = [30, 48]

# Additional attributes 
iscentre = np.zeros((NATOMS*NVESICLES))
indexcentre = np.zeros((NATOMS*NVESICLES))
phi_production = np.zeros((NATOMS*NVESICLES))
localmobility = np.zeros((NATOMS*NVESICLES))
localrange = np.zeros((NATOMS*NVESICLES))
nbonds2 = np.zeros((NATOMS*NVESICLES))

C2 = np.zeros((NATOMS, NBONDS2))
C2[0] = LINKS2
C2[10][0] = 1
C2[20][0] = 1
C2[30][0] = 1
C2[40][0] = 1
C2[50][0] = 1
C2[60][0] = 1

nbonds2[0] = 6
for link in LINKS2:
  nbonds2[link-1] = 1

localrange[::] = 4.0
localrange[-1] = 6.0

def file_to_array(filename):
  index, x, y, z, neighbours = [], [], [], [], []

  f = open(filename, "r")
  for line in f.readlines()[2:]:
    index.append(line.split()[4])
    x.append(line.split()[1])
    y.append(line.split()[2])
    z.append(line.split()[3])

    neighbour = list(map(int, line.split()[4:]))
    neighbours.append(neighbour)
    index = list(map(int, index))
    x = list(map(float, x))
    y = list(map(float, y))
    z = list(map(float, z))
  coords = np.concatenate(([x],[y],[z]))
  connectivity_matrix = neighbours
  return(np.array(index), np.array(coords), np.array(connectivity_matrix))

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
  

index, xyz, C = file_to_array("fullerene.xyz")
xyz = [normalize(xyz[index], -1, 1) for index in range(len(xyz))]

pair_distance = np.sqrt(
		(xyz[0][1]-xyz[0][2])**2 + 
		(xyz[1][1]-xyz[1][2])**2 +
		(xyz[2][1]-xyz[2][2])**2
		)

factor = BOND_LENGTH / pair_distance
xyz = rescale(xyz, factor)

for i in range(61):
  dist = np.sqrt(
		(xyz[0][0]-xyz[0][i])**2 + 
		(xyz[1][0]-xyz[1][i])**2 +
		(xyz[2][0]-xyz[2][i])**2
		)
  print(dist)


## Ideally duplicating coordinates as many times as NVESICLES 

# Indexes should start at 1
index = np.tile(index, NVESICLES)
index += 1

C = np.tile(C, (NVESICLES,1))
C += 1

xyz = np.tile(xyz, NVESICLES)

for n in range(1,NVESICLES+1):
  iscentre[(n-1)*NATOMS] = 1 #0, NATOMS, etc...
  indexcentre[(n-1)*NATOMS:1+n*NATOMS] = (n-1)*NATOMS+1 #(0, NATOMS-1), (NATOMS, 2*NATOMS), etc..
  phi_production[(n-1)*NATOMS] = 0.01
  localmobility[::] = 0.0
  localmobility[-1] = 0.5

  index[(n-1)*NATOMS:(n)*NATOMS] += (n-1)*NATOMS
  
  
  xyz[0, (n-1)*NATOMS:n*NATOMS] += XSHIFT[n-1]
  xyz[1, (n-1)*NATOMS:n*NATOMS] += YSHIFT[n-1]
  xyz[2, (n-1)*NATOMS:n*NATOMS] += ZSHIFT[n-1]

  
  C[(n-1)*NATOMS:n*NATOMS:] += (n-1)*NATOMS
  C[(n-1)*NATOMS] = np.zeros(NBONDS+1)

table = np.column_stack((index, xyz.T, C.T[1:].T, iscentre.T, indexcentre.T, phi_production.T, localrange.T, localmobility.T, nbonds2, C2))
np.savetxt("latticeFullerene.txt", table, fmt = '%3d %4f %4f %4f %3d %3d %3d %3d %3d %4f %4f %4f %1d %1d %1d %1d %1d %1d %1d')

