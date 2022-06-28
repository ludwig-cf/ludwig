########################################################################
#                                                                      #
#  extract.py                                                          #
#                                                                      #
#  Script for batch processing of data files in vtk-format             #
#  for visualisation in Paraview.                                      #
#                                                                      #
#  Requires executable 'extract' with corresponding flags set          #
#
#  Usage: $> python extract.py [options]
#
#  Options:
#    -v    velocity
#    -p    Compositional order parameter phi
#    -q    Q-tensor order parameter
#    -e    psi
#    -f    free energy density
#    -s    squirmer/polymer output
#    -t    squirmer/polymer/velocity output
#    -y    colloid output
#    -z    colloid velocity output
#                                                                      #
#  Edinburgh Soft Matter and Statistical Physics Group                 #
#  Edinburgh Parallel Computing Centre                                 #
#  University of Strathclyde, Glasgow, UK                              #
#                                                                      #
#  Contributing authors:                                               #
#  Kevin Stratford (kevin@epcc.ed.ac.uk)                               #
#  Oliver Henrich  (oliver.henrich@strath.ac.uk)                       #
#                                                                      #
#
#  (c) 2011-2020 The University of Edinburgh
#                                                                      #
########################################################################

import os
import sys
import getopt

nstart=1000   # Start timestep
nend=10000	# End timestep
nint=1000	# Increment

ngroup=1	# Number of output groups

a0_squ  = None  # the radius of squirmer
a0_poly = None  # the radius of monomer 

vel=False	# Switch for velocity 
q=False		# Switch for Q-tensor
phi=True	# Switch for binary fluid
mask=True	# Switch for binary fluid
psi=False	# Switch for electrokinetics
fed=False	# Switch for free energy
colcds=False	# Switch for colloid coordinate
colcdsvel=True # Switch for colloid coordinate and lattice velocity

squ_poly_cds = False     # Squirmer, polymer co-ordinate
squ_poly_cdsvel = False  # Squirmer, polymer co-ordinate; velocity 

try:
  opts, args = getopt.getopt(sys.argv[1:], "pmvc", ["nstart =", "nend =", "nint =",])
except:
  print("Error")

for opt, arg in opts:
        if (opt == "--nstart "):
          nstart = int(arg)
        elif (opt == "--nend "):
          nend = int(arg)
        elif (opt == "--nint "):
          nint = int(arg)
        elif (opt == "-p"):
          phi = True
        elif (opt == "-m"):
          mask = True
        elif (opt == "-v"):
          vel = True
        elif (opt == "-c"):
          colcdsvel = True

print("Extracting ", end = "")
if phi == True: print("phi fields, ", end = "")
if vel == True: print("velocity fields, ", end = "")
if mask == True: print("mask fields, ", end = "")
if colcdsvel == True: print("and colloid coordinates and velocities ", end = "")
print("from " + str(nstart) + " to " + str(nend) + " with increment " + str(nint))

# Set lists for analysis
metafile=[]
filelist=[]

if vel:
	metafile.append('vel.00%d-001.meta' % ngroup)
	filelist.append('filelist_vel')
	for i in range(nstart,nend+nint,nint):
          os.system('ls -t1 vel-%08.0d.00%d-001 >> filelist_vel' % (i,ngroup))

if phi:
	metafile.append('phi.%03.0d-001.meta' % ngroup)
	filelist.append('filelist_phi')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 phi-%08.0d.%03.0d-001 >> filelist_phi' % (i,ngroup))

if mask:
	metafile.append('flux_mask.%03.0d-001.meta' % ngroup)
	filelist.append('filelist_mask')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 flux_mask-%08.0d.%03.0d-001 >> filelist_mask' % (i,ngroup))

if colcdsvel==1:
	metafile.append('')
	filelist.append('filelist_colloid')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 config.cds%08.0d.001-001 >> filelist_colloid' % i)

# Create vtk-files
for i in range(len(filelist)):
	if filelist[i] == 'filelist_vel' or filelist[i] == 'filelist_phi' or filelist[i] == 'filelist_mask':
		datafiles=open(filelist[i],'r') 

		while 1:
			line=datafiles.readline()
			if not line: break

			#print(('\n# Processing %s' % line)) 

			stub=line.split('.',1)
			os.system('./extract -a -k %s %s' % (metafile[i],stub[0]))

		datafiles.close

	if filelist[i] == 'filelist_colloid':
		datafiles=open(filelist[i],'r') 

		while 1:
			line=datafiles.readline()
			if not line: break

			#print(('\n# Processing %s' % line))

			stub=line.split('.',2)
			datafilename = ('%s.%s' % (stub[0], stub[1]))
			outputfilename1 = ('col-%s.csv' % stub[1])
			outputfilename2 = ('velcol-%s.vtk' % stub[1])

			if colcds:
				os.system('./extract_colloids %s %d %s' % (datafilename,ngroup,outputfilename1))
			if colcdsvel:
				os.system('./extract_colloids %s %d %s %s' % (datafilename,ngroup,outputfilename1,outputfilename2))	

os.system('rm filelist*')

print('\n EXTRACTION COMPLETE \n')
