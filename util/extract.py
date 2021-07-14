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

nstart=1000	# Start timestep
nint=1000	# Increment
nend=10000	# End timestep
ngroup=1	# Number of output groups

a0_squ  = None  # the radius of squirmer
a0_poly = None  # the radius of monomer 

vel=False	# Switch for velocity 
q=False		# Switch for Q-tensor
phi=False	# Switch for binary fluid
psi=False	# Switch for electrokinetics
fed=False	# Switch for free energy
colcds=False	# Switch for colloid coordinate
colcdsvel=False	# Switch for colloid coordinate and lattice velocity

squ_poly_cds = False     # Squirmer, polymer co-ordinate
squ_poly_cdsvel = False  # Squirmer, polymer co-ordinate; velocity 

opts, args = getopt.getopt(sys.argv[1:], "vqpefyz")
for opt, arg in opts:
	if (opt == "v"):
		vel = True
	elif (opt == "q"):
		q = True
	elif (opt == "p"):
		phi = True
	elif (opt == "e"):
		psi = True
	elif (opt == "f"):
		fed = True
        elif (opt == "s"):
		squ_poly_cds = True
	elif (opt == "t"):
		squ_poly_cdsvel = True
	elif (opt == "y"):
		colcds = True
	elif (opt == "z"):
		colcdsvel = True
	else:
		sys.stdout.write("Bad argument\n")

# Set lists for analysis
metafile=[]
filelist=[]

if vel:
	metafile.append('vel.00%d-001.meta' % ngroup)
	filelist.append('filelist_vel')
	os.system('rm filelist_vel')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 vel-%08.0d.00%d-001 >> filelist_vel' % (i,ngroup))

if q:
	metafile.append('q.00%d-001.meta' % ngroup)
	filelist.append('filelist_q')
	os.system('rm filelist_q')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 q-%08.0d.00%d-001 >> filelist_q' % (i,ngroup))

if phi:
	metafile.append('phi.%03.0d-001.meta' % ngroup)
	filelist.append('filelist_phi')
	os.system('rm filelist_phi')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 phi-%08.0d.%03.0d-001 >> filelist_phi' % (i,ngroup))

if psi:
        metafile.append('psi.%03.0d-001.meta' % ngroup)
        filelist.append('filelist_psi')
        os.system('rm filelist_psi')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 psi-%08.0d.%03.0d-001 >> filelist_psi' % (i,ngroup))

if fed:
        metafile.append('fed.%03.0d-001.meta' % ngroup)
        filelist.append('filelist_fed')
        os.system('rm filelist_fed')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 fed-%08.0d.%03.0d-001 >> filelist_fed' % (i,ngroup))

if (colcds==1) or (colcdsvel==1):
	metafile.append('')
	filelist.append('filelist_colloid')
	os.system('rm filelist_colloid')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 config.cds%08.0d.001-001 >> filelist_colloid' % i)

if (squ_poly_cds==1) or (squ_poly_cdsvel==1):
        metafile.append('')
        filelist.append('filelist_squ_poly')
        os.system('rm filelist_squ_poly')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 config.cds%08.0d.001-001 >> filelist_squ_poly' % (i))

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

			if colcds:
				os.system('./extract_colloids %s %d %s' % (datafilename,ngroup,outputfilename1))
			if colcdsvel:
				os.system('./extract_colloids %s %d %s %s' % (datafilename,ngroup,outputfilename1,outputfilename2))	

		datafiles.close


        if filelist[i] == 'filelist_squ_poly':
                datafiles=open(filelist[i],'r') 

                while 1:
                        line=datafiles.readline()
                        if not line: break

                        print(('\n# Processing %s' % line))

                        stub=line.split('.',2)
                        datafilename = ('%s.%s' % (stub[0], stub[1]))
                        outputfilename1 = ('squ-%s.csv' % stub[1])
                        outputfilename2 = ('poly-%s.csv' % stub[1])
                        outputfilename3 = ('velsqu-%s.vtk' % stub[1])

                        if squ_poly_cds==1:
                                os.system('./extract_squirmer_polymer %s %d %s %s %f %f' % (datafilename,ngroup,outputfilename1,outputfilename2,a0_squ,a0_poly))
                        if squ_poly_cdsvel==1:
                                os.system('./extract_squirmer_polymer %s %d %s %s %f %f %s' % (datafilename,ngroup,outputfilename1,outputfilename2,a0_squ,a0_poly,outputfilename3))

		datafiles.close


os.system('rm filelist*')

print('# Done')
