# selects data columns from a file
import sys, os, re, math

x=int(sys.argv[1])-1
#y=int(sys.argv[2])-1
#z=int(sys.argv[3])-1

#ux=int(sys.argv[4])-1
#uy=int(sys.argv[5])-1
#uz=int(sys.argv[6])-1

datafilename=sys.argv[2]

# inputfiles
if datafilename=='filelist':
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
#			if line[0]=='#': # header sign
#				continue
			datastring=line.split()

			for a in range(len(datastring)):
				if a==x:
				       xdata=float(datastring[x])
#				if a==y:
#				       ydata=float(datastring[y])
#				if a==z:
#				       zdata=float(datastring[z])
#				if a==ux:
#				       uxdata=float(datastring[ux])
#				if a==uy:
#				       uydata=float(datastring[uy])
#				if a==uz:
#				       uzdata=float(datastring[uz])


#			unorm=uxdata*uxdata+uydata*uydata+uzdata*uzdata
#			if unorm>umax: 
#				umax=unorm
#			dataline=[xdata,ydata,zdata,uxdata,uydata,uzdata]
			out.write('%le\n' % xdata)
#			out.write('%d %d %d %le %le %le\n' % (dataline[0],dataline[1],dataline[2],dataline[3],dataline[4],dataline[5]))

		out.close
		file.close
#		print '%le %s' % (math.sqrt(umax),datafilename)
#		sys.stdout.flush()
	datafilenames.close
else:
	outputfilename= datafilename + '-sel'

	file=open(datafilename,'r')
	out=open(outputfilename,'w')

	dataline=[]
	data=[]

	while 1:


		line=file.readline()
		if not line: break
#		if line[0]=='#': # header sign
#			continue
		datastring=line.split()

		for a in range(len(datastring)):
			if a==x:
			       xdata=float(datastring[x])
#			if a==y:
#			       ydata=float(datastring[y])
#			if a==z:
#			       zdata=float(datastring[z])
#			if a==ux:
#			       uxdata=float(datastring[ux])
#			if a==uy:
#			       uydata=float(datastring[uy])
#			if a==uz:
#			       uzdata=float(datastring[uz])


#		unorm=uxdata*uxdata+uydata*uydata+uzdata*uzdata
#		if unorm>umax: 
#			umax=unorm
#		dataline=[xdata,ydata,zdata,uxdata,uydata,uzdata]
		out.write('%g\n' % xdata)
#		out.write('%d %d %d %le %le %le\n' % (dataline[0],dataline[1],dataline[2],dataline[3],dataline[4],dataline[5]))

	out.close
	file.close
#	print '%le %s' % (math.sqrt(umax),datafilename)
#	sys.stdout.flush()
