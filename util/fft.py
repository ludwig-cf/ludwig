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
#  $Id: vtk_extract_script.py 2522 2014-11-05 15:20:06Z ohenrich $$    #
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
nend=100000	# End timestep

phi=1		# Switch for binary fluid

# Set lists for analysis
filelist=[]

if phi==1:
	filelist.append('filelist_phi')
	os.system('rm filelist_phi')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 phi-%08.0d.vtk >> filelist_phi' % i)

os.system('g++ -o fft fft.cc -lfftw3 -lm')

for i in range(len(filelist)):
	datafiles=open(filelist[i],'r') 

	while 1:
		line=datafiles.readline()
		if not line: break

#		print '\n# Processing %s' % line 

		os.system('./fft %s' % line)

		datafiles.close

os.system('rm filelist*')

#print('# Done')
