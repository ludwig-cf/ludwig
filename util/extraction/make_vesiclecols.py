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
  vesicle_fphi = np.zeros(3)
  vesicle_fsub = np.zeros(3)
  vesicle_fsprings = np.zeros(3)
  vesicle_tphi = np.zeros(3)
  vesicle_tsprings = np.zeros(3)
  vesicle_total_force = np.zeros(3)
  vesicle_total_torque = np.zeros(3)

  new_lines = []
  new_lines.append('vesicle_fphix,vesicle_fphiy,vesicle_fphiz,vesicle_fsubx,vesicle_fsuby,vesicle_fsubz,vesicle_fspringsx,vesicle_fspringsy,vesicle_fspringsz,vesicle_tphix,vesicle_tphiy,vesicle_tphiz,vesicle_tspringsx,vesicle_tspringsy,vesicle_tspringsz,vesicle_total_forcex,vesicle_total_forcey,vesicle_total_forcez,vesicle_total_torquex,vesicle_total_torquey,vesicle_total_torquez')

  with open(filelist_col[i], 'r') as inputfile:
    original_lines = inputfile.readlines()

    for line in original_lines[1::]:
      vesicle_fphi[0] += float(line.split(',')[14]) 
      vesicle_fphi[1] += float(line.split(',')[15]) 
      vesicle_fphi[2] += float(line.split(',')[16]) 

      vesicle_fsub[0] += float(line.split(',')[17]) 
      vesicle_fsub[1] += float(line.split(',')[18]) 
      vesicle_fsub[2] += float(line.split(',')[19]) 

      vesicle_fsprings[0] += float(line.split(',')[20]) 
      vesicle_fsprings[1] += float(line.split(',')[21]) 
      vesicle_fsprings[2] += float(line.split(',')[22]) 

      vesicle_tphi[0] += float(line.split(',')[23]) 
      vesicle_tphi[1] += float(line.split(',')[24]) 
      vesicle_tphi[2] += float(line.split(',')[25]) 
      
      vesicle_fsprings[0] += float(line.split(',')[26]) 
      vesicle_fsprings[1] += float(line.split(',')[27]) 
      vesicle_fsprings[2] += float(line.split(',')[28]) 

      vesicle_total_force[0] += float(line.split(',')[29]) 
      vesicle_total_force[1] += float(line.split(',')[30]) 
      vesicle_total_force[2] += float(line.split(',')[31]) 

      vesicle_total_torque[0] += float(line.split(',')[32]) 
      vesicle_total_torque[1] += float(line.split(',')[33]) 
      vesicle_total_torque[2] += float(line.split(',')[34]) 

  new_lines.append('{:7e}'.format(vesicle_fphi[0]) +     ',    '
                 + '{:7e}'.format(vesicle_fphi[1]) +     ',    ' 
                 + '{:7e}'.format(vesicle_fphi[2]) +     ',    '

                 + '{:7e}'.format(vesicle_fsub[0]) +     ',    '
                 + '{:7e}'.format(vesicle_fsub[1]) +     ',    ' 
                 + '{:7e}'.format(vesicle_fsub[2]) +     ',    '

                 + '{:7e}'.format(vesicle_fsprings[0]) +     ',    '
                 + '{:7e}'.format(vesicle_fsprings[1]) +     ',    ' 
                 + '{:7e}'.format(vesicle_fsprings[2]) +     ',    '

                 + '{:7e}'.format(vesicle_tphi[0]) +     ',    '
                 + '{:7e}'.format(vesicle_tphi[1]) +     ',    ' 
                 + '{:7e}'.format(vesicle_tphi[2]) +     ',    '
 
                 + '{:7e}'.format(vesicle_tsprings[0]) +     ',    '
                 + '{:7e}'.format(vesicle_tsprings[1]) +     ',    ' 
                 + '{:7e}'.format(vesicle_tsprings[2]) +     ',    '
 
                 + '{:7e}'.format(vesicle_total_force[0]) +     ',    '
                 + '{:7e}'.format(vesicle_total_force[1]) +     ',    ' 
                 + '{:7e}'.format(vesicle_total_force[2]) +     ',    '

                 + '{:7e}'.format(vesicle_total_torque[0]) +     ',    '
                 + '{:7e}'.format(vesicle_total_torque[1]) +     ',    ' 
                 + '{:7e}'.format(vesicle_total_torque[2]))
 
  with open('vesicle'+filelist_col[i], 'w') as outputfile:
    for line in new_lines:
      outputfile.write(line)
      outputfile.write("\n")

if os.path.exists("filelist"): 
  os.remove("filelist")
