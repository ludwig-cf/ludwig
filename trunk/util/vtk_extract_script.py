########################################################################
#			   					       #
#  vtk_extract_script.py 					       #
#			   					       #
#  Script for creating data files in vtk-format for visualisation in   #
#  Paraview.						               #
#  Requires vtk_extract.c with corresponding flags set and	       #
#  an executable 'extract_colloids' for colloid processing. 	       #				
#								       #				
#  Usage: $> python vtk_extract_script.py			       #
#								       #
#  $Id$$							       #
#								       #	
#  Edinburgh Soft Matter and Statistical Physics Group and	       #
#  Edinburgh Parallel Computing Centre				       #
#								       #	
#  Oliver Henrich (ohenrich@epcc.ed.ac.uk)			       #
#  (c) 2014 The University of Edinburgh				       #
#			   					       #
########################################################################

import sys, os, re, math

nstart=1000	# Start timestep
nint=1000	# Increment
nend=40000	# End timestep
ngroup=1	# Number of output groups

vel=1		# Switch for velocity 
q=1		# Switch for Q-tensor postprocessing
phi=1		# Switch for binary fluid
psi=1		# Switch for electrokinetics
fed=1		# Switch for free energy
colloid=1	# Switch for colloid postprocessing

# Set lists for analysis
type=[]
x=[]
y=[]
z=[]
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

if colloid==1:
	metafile.append('')
	filelist.append('filelist_colloid')
	os.system('rm filelist_colloid')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 config.cds%08.0d.001-001 >> filelist_colloid' % i)

# Create vtk-files
for i in range(len(filelist)):
	if filelist[i] != 'filelist_colloid':
		datafiles=open(filelist[i],'r') 

		print('# Creating vtk-datafiles')

		while 1:
			line=datafiles.readline()
			if not line: break

			print '\n# Processing %s' % line 

			stub=line.split('.',1)
			os.system('./vtk_extract %s %s' % (metafile[i],stub[0]))

		datafiles.close

	if filelist[i] == 'filelist_colloid':
		datafiles=open(filelist[i],'r') 

		print('# Creating csv-files')

		while 1:
			line=datafiles.readline()
			if not line: break

			print '# Processing %s' % line 

			os.system('./extract_colloids %s %d %s' % (datafilename,ngroup,outputfilename))
	
		datafilenames.close

os.system('rm filelist*')

print('# Done')
