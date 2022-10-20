import numpy as np
import os 
import csv
import sys

# create file list with names of files to loop over (from col-cds00001000.csv to col-cds00XXXXXX.csv)
# loop over list and open file
# open col-cds file
# calculate totfphi
# write in new file

nstart = 1000
nend = 1000000
nint = 10000

args = sys.argv[1:]
nstart = int(args[0])
nend = int(args[1])
nint = int(args[2])

for i in range(nstart,nend+nint,nint):
  iteration = int(i/nint) - 1
  new_lines = []
  new_lines.append('totalx,totaly,totalz,fluidx,fluidy,fluidz,colloidx,colloidy,colloidz')

  with open('Momentum.txt', 'r') as momentumfile:
    original_lines = momentumfile.readlines()
    line = original_lines[iteration + 1]

    totalx = float(line.split(',')[0])
    totaly = float(line.split(',')[1])
    totalz = float(line.split(',')[2])
    
    fluidx = float(line.split(',')[3])
    fluidy = float(line.split(',')[4])
    fluidz = float(line.split(',')[5])
 
    colloidx = float(line.split(',')[6])
    colloidy = float(line.split(',')[7])
    colloidz = float(line.split(',')[8])

  new_lines.append('{:7e}'.format(totalx) + ',    ' + '{:7e}'.format(totaly) + ',    ' + '{:7e}'.format(totalz) + ',    ' + '{:7e}'.format(fluidx) + ',    ' + '{:7e}'.format(fluidy) + ',    ' + '{:7e}'.format(fluidz) + ',    ' + '{:7e}'.format(colloidx) + ',    ' + '{:7e}'.format(colloidy) + ',    ' + '{:7e}'.format(colloidz))
 
  filename = 'momentum-%08.0d.csv' % i
  with open(filename, 'w') as outputfile:
    for line in new_lines:
      outputfile.write(line)
      outputfile.write("\n")

if os.path.exists("filelist"): 
  os.remove("filelist")
