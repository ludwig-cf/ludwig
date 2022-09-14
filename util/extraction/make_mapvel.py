import numpy as np
import os 
import csv
import sys

# Open velocity files
# Find where vel is null
# Assign node to colloid node in another array
# Output mapcol file
# Next vel

nstart = 1000
nend = 1000000
nint = 10000

args = sys.argv[1:]
nstart = int(args[0])
nend = int(args[1])
nint = int(args[2])

filelist = []
filelist_vel = []

for i in range(nstart,nend+nint,nint):
  os.system('ls -t1 vel-%08.0d.vtk >> filelist' % i)
  filelist_vel.append('vel-%08.0d.vtk' % i)

for i in range(len(filelist_vel)):

  with open(filelist_vel[i], 'r') as inputfile:
    original_lines = inputfile.readlines()
    HEADERS = original_lines[:9]
    DIMENSIONS = HEADERS[4]
    NX, NY, NZ = int(DIMENSIONS.split()[1]), int(DIMENSIONS.split()[2]), int(DIMENSIONS.split()[3]) 

    COLMAP = []
    for heads in HEADERS:
      COLMAP.append(heads.strip())
    COLMAP[8] = "SCALARS mapcol int 1"
    COLMAP.append("LOOKUP_TABLE default")


    DATA = np.array(original_lines[9:])
    n = 0

    for index in range(NX*NY*NZ):
      kl = index // NX*NY
      jl = (index - kl*NX*NY) // NX
      il = index - jl*NX - kl*NX*NY
      vels = DATA[index].split()
         
      if vels[0] == "0.000000e+00" and vels[1] == "0.000000e+00" and vels[2] == "0.000000e+00":
        COLMAP.append("1")
        n += 1
      else: 
        COLMAP.append("0")

  print(n)


  with open('map_'+filelist_vel[i], 'w') as outputfile:
    for line in COLMAP:
      outputfile.write(line)
      outputfile.write("\n")

if os.path.exists("filelist"): 
  os.remove("filelist")
