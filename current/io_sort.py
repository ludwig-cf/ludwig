import sys, os, re, math

# starts the io_sort routine on a filelist
# NOTE: correct system size has to be given in io_sort.c


# command line input is

# 1 number of files in grouping
# 2 list of file stubs to process



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

