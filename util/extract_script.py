# Script for creating ASCII data file with io_sort.c
# and creating vtk-data files for visualization

# input as follows: 
# size; NOTE: correct system size has to be also given in io_sort.c 
# type: 1 scalar, 2 vector
# no of files in output grouping
# selected columns for output into vtk-file starting with 1
# filelist name (default filelist)

import sys, os, re, math
Lx=32
Ly=32
Lz=32

nstart=1000
nint=1000
nend=1000
ngroup=1

op=0
vel=0
dir=0
phi=0
biaxop=0
colloid=0
psi=0
fed=0
bfed=0
gfed=0

create_ascii_file=1
create_paraview_file=1

# collect and set lists for analysis
type=[]
x=[]
y=[]
z=[]
metafile=[]
filelist=[]

if op==1:
	type.append('1')
	x.append('1')
	y.append('0')
	z.append('0')
	metafile.append('qs_dir.00%d-001.meta' % ngroup)
	filelist.append('filelist_op')
	os.system('rm filelist_op')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 qs_dir-%08.0d.00%d-001 >> filelist_op' % (i,ngroup))

if vel==1:
	type.append('2')
	x.append('1')
	y.append('2')
	z.append('3')
	metafile.append('vel.00%d-001.meta' % ngroup)
	filelist.append('filelist_vel')
	os.system('rm filelist_vel')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 vel-%08.0d.00%d-001 >> filelist_vel' % (i,ngroup))

if dir==1:
	type.append('3')
	x.append('2')
	y.append('3')
	z.append('4')
	metafile.append('qs_dir.00%d-001.meta' % ngroup)
	filelist.append('filelist_dir')
	os.system('rm filelist_dir')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 qs_dir-%08.0d.00%d-001 >> filelist_dir' % (i,ngroup))

if phi==1:
	type.append('4')
	x.append('1')
	y.append('0')
	z.append('0')
	metafile.append('phi.%03.0d-001.meta' % ngroup)
	filelist.append('filelist_phi')
	os.system('rm filelist_phi')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 phi-%08.0d.%03.0d-001 >> filelist_phi' % (i,ngroup))

if biaxop==1:
	type.append('5')
	x.append('5')
	y.append('0')
	z.append('0')
	metafile.append('qs_dir.00%d-001.meta' % ngroup)
	filelist.append('filelist_biaxop')
	os.system('rm filelist_biaxop')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 qs_dir-%08.0d.00%d-001 >> filelist_biaxop' % (i,ngroup))

if psi==1:
        type.append('6')
        x.append('1')
        y.append('0')
        z.append('0')
        metafile.append('psi.%03.0d-001.meta' % ngroup)
        filelist.append('filelist_psi')
        os.system('rm filelist_psi')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 psi-%08.0d.%03.0d-001 >> filelist_psi' % (i,ngroup))

if fed==1:
        type.append('7')
        x.append('1')
        y.append('0')
        z.append('0')
        metafile.append('fed.%03.0d-001.meta' % ngroup)
        filelist.append('filelist_fed')
        os.system('rm filelist_fed')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 fed-%08.0d.%03.0d-001 >> filelist_fed' % (i,ngroup))

if bfed==1:
        type.append('8')
        x.append('2')
        y.append('0')
        z.append('0')
        metafile.append('fed.%03.0d-001.meta' % ngroup)
        filelist.append('filelist_bfed')
        os.system('rm filelist_bfed')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 fed-%08.0d.%03.0d-001 >> filelist_bfed' % (i,ngroup))

if gfed==1:
        type.append('9')
        x.append('3')
        y.append('0')
        z.append('0')
        metafile.append('fed.%03.0d-001.meta' % ngroup)
        filelist.append('filelist_gfed')
        os.system('rm filelist_gfed')
        for i in range(nstart,nend+nint,nint):
                os.system('ls -t1 fed-%08.0d.%03.0d-001 >> filelist_gfed' % (i,ngroup))

os.system('gcc -o extract extract.c -lm')

if colloid==1:
      	type.append('99')
	x.append('0')
	y.append('0')
	z.append('0')
	metafile.append('')
	filelist.append('filelist_colloid')
	os.system('rm filelist_colloid')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 config.cds%08.0d.001-001 >> filelist_colloid' % i)


# create ASCII datafile
if create_ascii_file==1:
	for i in range(len(type)):
		if type[i] != '99':
			datafiles=open(filelist[i],'r') 

			print('# creating datafiles')

			while 1:
				line=datafiles.readline()
				if not line: break

				print '# processing %s' % line 

				stub=line.split('.',1)
				os.system('./extract %s %s' % (metafile[i],stub[0]))

			datafiles.close

if create_paraview_file==1:

	# create datafile list
	if op==1:
	        os.system('rm filelist_op')
		for i in range(nstart,nend+nint,nint):
				os.system('ls -t1 qs_dir-%08.0d >> filelist_op' % i)
	if vel==1:
	        os.system('rm filelist_vel')
		for i in range(nstart,nend+nint,nint):
				os.system('ls -t1 vel-%08.0d >> filelist_vel' % i)
	if dir==1:
	        os.system('rm filelist_dir')
		for i in range(nstart,nend+nint,nint):
				os.system('ls -t1 qs_dir-%08.0d >> filelist_dir' % i)
	if phi==1:
	        os.system('rm filelist_phi')
		for i in range(nstart,nend+nint,nint):
				os.system('ls -t1 phi-%08.0d >> filelist_phi' % i)
	if biaxop==1:
	        os.system('rm filelist_biaxop')
		for i in range(nstart,nend+nint,nint):
				os.system('ls -t1 qs_dir-%08.0d >> filelist_biaxop' % i)
        if psi==1:
                os.system('rm filelist_psi')
                for i in range(nstart,nend+nint,nint):
                                os.system('ls -t1 psi-%08.0d >> filelist_psi' % i)
        if fed==1:
                os.system('rm filelist_fed')
                for i in range(nstart,nend+nint,nint):
                                os.system('ls -t1 fed-%08.0d >> filelist_fed' % i)
        if bfed==1:
                os.system('rm filelist_bfed')
                for i in range(nstart,nend+nint,nint):
                                os.system('ls -t1 fed-%08.0d >> filelist_bfed' % i)
        if gfed==1:
                os.system('rm filelist_gfed')
                for i in range(nstart,nend+nint,nint):
                                os.system('ls -t1 fed-%08.0d >> filelist_gfed' % i)

	# create vtk-header
	for i in range(len(type)):

		x[i]=int(x[i])-1
		y[i]=int(y[i])-1
		z[i]=int(z[i])-1

		headerlines=[]

		if type[i]!='99':

			headerlines.append('# vtk DataFile Version 2.0')
			headerlines.append('Generated by create_paraview_file')
			headerlines.append('ASCII')
			headerlines.append('DATASET STRUCTURED_POINTS')
			headerlines.append('DIMENSIONS  %d %d %d' %(Lx,Ly,Lz))
			headerlines.append('ORIGIN 0 0 0')
			headerlines.append('SPACING 1 1 1')
			headerlines.append('POINT_DATA %d' %(Lx*Ly*Lz))
		if type[i]=='1' or type[i]=='4' or type[i]=='5' or type[i]=='6' or type[i]=='7' or type[i]=='8' or type[i]=='9':
			headerlines.append('SCALARS scalar%d float 1' %i)
			headerlines.append('LOOKUP_TABLE default')
		if type[i]=='2' or type[i]=='3':
			headerlines.append('VECTORS velocity float')

		print('# creating paraview-files')

		# inputfiles
		datafilenames=open(filelist[i],'r')

		while 1:
			line=datafilenames.readline()
			if not line: break

			if type[i]=='1':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-op.vtk'

			if type[i]=='2':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-velo.vtk'

			if type[i]=='3':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-dir.vtk'

			if type[i]=='4':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-phi.vtk'

			if type[i]=='5':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-biaxop.vtk'

			if type[i]=='6':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-psi.vtk'

			if type[i]=='7':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-fed.vtk'

			if type[i]=='8':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-bfed.vtk'

			if type[i]=='9':
				linestring=line.split()
				datafilename=linestring[0]
				outputfilename= datafilename + '.dat-gfed.vtk'

			if type[i]=='99':
				linestring=line.split()
				datafilename=linestring[0]
				linestring=line.split('.')
				outputfilename= 'colloid-' + linestring[1] + '.csv'

			print '# processing %s' % outputfilename

	                if type[i]!='99':
				file=open(datafilename,'r')
				out=open(outputfilename,'w')

				dataline=[]
				data=[]

				# write header
				for j in range(len(headerlines)):
					out.write('%s\n' % headerlines[j]) 


				while 1:

			     		line=file.readline()
					if not line: break

					datastring=line.split()

					if type[i]=='1' or type[i]=='4' or type[i]=='5' or type[i]=='6' or type[i]=='7' or type[i]=='8' or type[i]=='9':
						xdata=float(datastring[x[i]])
#						if abs(xdata)<1e-25: xdata=1e-25
						out.write('%12.5le\n' % xdata)
					if type[i]=='2' or type[i]=='3':
						xdata=float(datastring[x[i]])
						ydata=float(datastring[y[i]])
						zdata=float(datastring[z[i]])
						out.write('%.5le %.5le %.5le\n' % (xdata,ydata,zdata))
		
	       			out.close
				file.close

			if type[i]=='99':
				os.system('./extract_colloids %s %s' % (datafilename,outputfilename))

		datafilenames.close

os.system('rm filelist*')

print('# done')
