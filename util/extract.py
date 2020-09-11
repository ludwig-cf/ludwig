########################################################################
#                                                                      #
#  extract.py                                                          #
#                                                                      #
#  Script for batch processing of data files in vtk-format             #
#  for visualisation in Paraview.                                      #
#                                                                      #
#  Requires executable 'extract' with corresponding flags set          #
#
#  Usage: $> python extract.py                                         #
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
#  (c) 2011-2018 The University of Edinburgh                           #
#                                                                      #
########################################################################

import sys, os, re, math

nstart=1000	# Start timestep
nint=1000	# Increment
nend=10000	# End timestep
ngroup=1	# Number of output groups

vel=1		# Switch for velocity 
q=1		# Switch for Q-tensor
phi=0		# Switch for binary fluid
psi=0		# Switch for electrokinetics
fed=0		# Switch for free energy
colcds=0	# Switch for colloid coordinate
colcdsvel=1	# Switch for colloid coordinate and lattice velocity

# Set lists for analysis
metafile=[]
filelist=[]

if vel==1:
	metafile.append('vel.00%d-001.meta' % ngroup)
	filelist.append('filelist_vel')
	os.system('rm filelist_vel')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 vel-%08.0d.00%d-001 >> filelist_vel' % (i,ngroup))

if q==1:
	metafile.append('q.00%d-001.meta' % ngroup)
	filelist.append('filelist_q')
	os.system('rm filelist_q')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 q-%08.0d.00%d-001 >> filelist_q' % (i,ngroup))

if phi==1:
	metafile.append('phi.%03.0d-001.meta' % ngroup)
	filelist.append('filelist_phi')
	os.system('rm filelist_phi')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 phi-%08.0d.%03.0d-001 >> filelist_phi' % (i,ngroup))

if psi==1:
        metafile.append('psi.%03.0d-001.meta' % ngroup)
        filelist.append('filelist_psi')
        os.system('rm filelist_psi')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 psi-%08.0d.%03.0d-001 >> filelist_psi' % (i,ngroup))

if fed==1:
        metafile.append('fed.%03.0d-001.meta' % ngroup)
        filelist.append('filelist_fed')
        os.system('rm filelist_fed')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 fed-%08.0d.%03.0d-001 >> filelist_fed' % (i,ngroup))

os.system('gcc -o vtk_extract vtk_extract.c -lm')

if (colcds==1) or (colcdsvel==1):
	metafile.append('')
	filelist.append('filelist_colloid')
	os.system('rm filelist_colloid')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 config.cds%08.0d.001-001 >> filelist_colloid' % i)

# Create vtk-files
for i in range(len(filelist)):
	if filelist[i] == 'filelist_vel' or filelist[i] == 'filelist_phi':
		datafiles=open(filelist[i],'r') 

		while 1:
			line=datafiles.readline()
			if not line: break

			print(('\n# Processing %s' % line)) 

			stub=line.split('.',1)
			os.system('./extract -a -k %s %s' % (metafile[i],stub[0]))

		datafiles.close

	if filelist[i] == 'filelist_q':
		datafiles=open(filelist[i],'r') 

		while 1:
			line=datafiles.readline()
			if not line: break

			print(('\n# Processing %s' % line)) 

			stub=line.split('.',1)
			os.system('./extract -a -k -s -d %s %s' % (metafile[i],stub[0]))

		datafiles.close

	if filelist[i] == 'filelist_colloid':
		datafiles=open(filelist[i],'r') 

		while 1:
			line=datafiles.readline()
			if not line: break

			print(('\n# Processing %s' % line))

			stub=line.split('.',2)
			datafilename = ('%s.%s' % (stub[0], stub[1]))
			outputfilename1 = ('col-%s.csv' % stub[1])
			outputfilename2 = ('velcol-%s.vtk' % stub[1])

			if colcds==1:
				os.system('./extract_colloids %s %d %s' % (datafilename,ngroup,outputfilename1))
			if colcdsvel==1:
				os.system('./extract_colloids %s %d %s %s' % (datafilename,ngroup,outputfilename1,outputfilename2))	

os.system('rm filelist*')

print('# Done')
