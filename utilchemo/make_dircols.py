import numpy as np
import os 
import csv
import sys

# create file list with names of files to loop over (from col-cds00001000.csv to col-cds00XXXXXX.csv)
# loop over list and open file
# open col-cds file
# calculate m 
# calculate m.v = cos(alpha)
# append m
# append alpha

nstart = 1000
nend = 1000000
nint = 10000
indexcentre = 1
indexortho = 27

args = sys.argv[1:]
nstart = int(args[0])
nend = int(args[1])
nint = int(args[2])

filelist = []
filelist_col = []

for i in range(nstart,nend+nint,nint):
  os.system('ls -t1 col-cds%08.0d.csv >> filelist' % i)
  filelist_col.append('col-cds%08.0d.csv' % i)
m, n, v, rc, ro = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
for i in range(len(filelist_col)):

  foundindexcentre = False
  foundindexortho = False

  with open(filelist_col[i], 'r') as inputfile:
    original_lines = inputfile.readlines()

    for line in original_lines[1::]:
      if (foundindexcentre == True): break
      index = float(line.split(',')[0])
      if (index == indexcentre): 
        foundindexcentre = True
        rc[0], rc[1], rc[2] = float(line.split(',')[1]), float(line.split(',')[2]), float(line.split(',')[3]) 

        m[0], m[1], m[2] = float(line.split(',')[4]), float(line.split(',')[5]), float(line.split(',')[6]) 

        n[0], n[1], n[2] = float(line.split(',')[7]), float(line.split(',')[8]), float(line.split(',')[9]) 

        v[0], v[1], v[2] = float(line.split(',')[10]), float(line.split(',')[11]), float(line.split(',')[12]) 


    normv = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    v /= normv

    mdotv = np.dot(m,v)
    if mdotv > 1.: mdotv = 1.
    if mdotv < -1.: mdotv = -1.
      
    alpha = np.arccos(mdotv)

    new_lines = []
    new_lines.append('x0, y0, z0, mx, my, mz, nx, ny, nz, alpha')
    new_lines.append('{:7e}'.format(rc[0]) + ',    ' + '{:7e}'.format(rc[1]) + ',    ' + '{:7e}'.format(rc[2]) + ',    ' + '{:7e}'.format(m[0]) + ',    ' + '{:7e}'.format(m[1]) + ',    ' + '{:7e}'.format(m[2]) + ',    ' + '{:7e}'.format(n[0]) + ',    ' + '{:7e}'.format(n[1]) + ',    ' + '{:7e}'.format(n[2]) + ',    ' + '{:7e}'.format(alpha))
 
  with open('dir'+filelist_col[i], 'w') as outputfile:
    for line in new_lines:
      outputfile.write(line)
      outputfile.write("\n")

if os.path.exists("filelist"): 
  os.remove("filelist")
