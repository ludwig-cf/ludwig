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

os.system('ls -t1 order_velo.*.*.dat > filelist')

x=11
#y=int(sys.argv[2])-1
#z=int(sys.argv[3])-1

#ux=int(sys.argv[4])-1
#uy=int(sys.argv[5])-1
#uz=int(sys.argv[6])-1

# inputfiles
datafilenames=open('filelist','r')

while 1:
       line=datafilenames.readline()

       if not line: break

       linestring=line.split()
       datafilename=linestring[0]

       print 'processing %s' % datafilename

       outputfilename= datafilename + '-sel'

       file=open(datafilename,'r')
       out=open(outputfilename,'w')

       dataline=[]
       data=[]

       while 1:


	       line=file.readline()
	       if not line: break
#                       if line[0]=='#': # header sign
#                               continue
	       datastring=line.split()

	       for a in range(len(datastring)):
		       if a==x:
			      xdata=float(datastring[x])
#                               if a==y:
#                                      ydata=float(datastring[y])
#                               if a==z:
#                                      zdata=float(datastring[z])
#                               if a==ux:


#                       dataline=[xdata,ydata,zdata,uxdata,uydata,uzdata]
	       out.write('%g\n' % xdata)
#                       out.write('%d %d %d %le %le %le\n' % (dataline[0],dataline[1],dataline[2],dataline[3],dataline[4],dataline[5]))

       out.close
       file.close
datafilenames.close
