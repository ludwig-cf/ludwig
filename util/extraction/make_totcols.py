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

filelist_col = []
for i in range(nstart,nend+nint,nint):
  os.system('ls -t1 col-cds%08.0d.csv >> filelist' % i)
  filelist_col.append('col-cds%08.0d.csv' % i)

for i in range(len(filelist_col)):
  totfphi = np.zeros(3)

  new_lines = []
  new_lines.append('totphi_box,totphi_vesicle,totfphix,totfphiy,totfphiz')

  with open('totphi.txt', 'r') as totphifile:
    original_lines = totphifile.readlines()
    line = original_lines[i]
    totphi_vesicle = float(line.split(',')[0])
    totphi_box = float(line.split(',')[1])
 
  with open(filelist_col[i], 'r') as inputfile:
    original_lines = inputfile.readlines()

    for line in original_lines[1::]:
      totfphi[0] += float(line.split(',')[14]) 
      totfphi[1] += float(line.split(',')[15]) 
      totfphi[2] += float(line.split(',')[16]) 

  new_lines.append('{:7e}'.format(totphi_box) + ',    ' + '{:7e}'.format(totphi_vesicle) + ',    ' + '{:7e}'.format(totfphi[0]) + ',    ' + '{:7e}'.format(totfphi[1]) + ',    ' + '{:7e}'.format(totfphi[2]))
 
  with open('tot'+filelist_col[i], 'w') as outputfile:
    for line in new_lines:
      outputfile.write(line)
      outputfile.write("\n")

if os.path.exists("filelist"): 
  os.remove("filelist")
