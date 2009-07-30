import sys, os, re, math

# io_sort routine
# NOTE: correct system size has to be given in io_sort.c


# command line input is

# 1 list of file stubs to process
# 2 number of files in grouping


filelist=sys.argv[1]
ngroup=int(sys.argv[2])

os.system('gcc -o io_sort io_sort.c -lm')

# datafiles
datafiles=open(filelist,'r') 

while 1:
	line=datafiles.readline()

	print '\n'

	if not line: break

	stub=line.split('-',1)

	filetype=stub[0].split('.',1)


	if filetype[0] == 'disc':
		antype=1
		cmd = './io_sort %d %d %s' % (antype,ngroup,stub[0])
		os.system(cmd)


	if filetype[0] == 'order_velo':
		antype=2
		cmd = './io_sort %d %d %s' % (antype,ngroup,stub[0])
		os.system(cmd)


	if filetype[0] == 'sig':
		antype=3
		cmd = './io_sort %d %d %s' % (antype,ngroup,stub[0])
		os.system(cmd)


	if filetype[0] == 'dir':
		antype=4
                cmd = './io_sort %d %d %s' % (antype,ngroup,stub[0])
		os.system(cmd)



datafiles.close


# selecting order parameter from 3D data file


#xdata=int(sys.argv[2])
#ydata=int(sys.argv[3])
#zdata=int(sys.argv[4])
#tdata=int(sys.argv[5])

xdata=1
ydata=2
zdata=3
tdata=12



os.system('ls -t1 order_velo.*.*.dat > filelist')


inputdatafiles=open('filelist','r')

while 1:
       line=inputdatafiles.readline()

       if not line: break
       linestring=line.split()
       datafile=linestring[0]

       outputfile= datafile + '-sel'

       file=open(datafile,'r')
       out=open(outputfile,'w')

       print '\n# selecting data form %s' % datafile
       print '# x / y / z / data from column # %d / %d / %d / %d' % (xdata,ydata,zdata,tdata)
       print '# writing data to file %s\n' % outputfile

       datalines=0
       data=[]

       while 1:
	       line=file.readline()
	       if not line: break
	       if line[0]=='#': # header sign
		       continue
	       datastring=line.split()
	       dataline=[]

       # selects x, y and z components from inputfile and appends them to the output data array

	       for j in range(len(datastring)):
		       if j==xdata-1:
			       dataline.append(float(datastring[xdata-1]))
	       for j in range(len(datastring)):
		       if j==ydata-1:
			       dataline.append(float(datastring[ydata-1]))
	       for j in range(len(datastring)):
		       if j==zdata-1:
			       dataline.append(float(datastring[zdata-1]))
	       for j in range(len(datastring)):
		       if j==tdata-1:
			       dataline.append(float(datastring[tdata-1]))

#              out.write('%d %d %d %f\n' % (dataline[0],dataline[1],dataline[2],dataline[3]))
	       out.write('%f\n' % dataline[3])

#              data.append(dataline)
	       datalines=datalines+1
       file.close


       # write output data array to output file
       # number of lines

#               out.write('%d\n' % datalines)

       # selected data
#               for i in range(len(data)):
       #       out.write('%d %d %d %f\n' % (data[i][0],data[i][1],data[i][2],data[i][3]))
#                       out.write('%f\n' % data[i][3])

       # if block data form is required

       #       if data[i][0] != data[i+1][0]:
       #               out.write('\n') 
	       #out.write('%12.8g %12.8g %12.8g %12.8g' % (data[len(data)-1][0],data[len(data)-1][1],data[len(data)-1][2],data[len(data)-1][3]))

       out.close

inputdatafiles.close


