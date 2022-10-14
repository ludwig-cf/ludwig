import os
import sys
import getopt

nstart=100    
nend= 5000	
nint=100

ngroup=1	# Number of output groups

vel =		1	
phi =		1	
temperature =	1	
advective_flux_psi=1
total_flux_psi=1
mu = 1
fed= 1
colcdsvel =	1	

q =		0	
psi =		0
fed =		0
colcds =	0

try:
  opts, args = getopt.getopt(sys.argv[1:], "ptvc", ["nstart =", "nend =", "nint =",])
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
        elif (opt == "-t"):
          temperature = True
        elif (opt == "-v"):
          vel = True
        elif (opt == "-c"):
          colcdsvel = True

        elif (opt == "-a"):
          advective_flux_psi = True
        elif (opt == "-tot"):
          total_flux_psi = True



print("Extracting ", end = "")
if phi == True: print("phi fields, ", end = "")
if vel == True: print("velocity fields, ", end = "")
if temperature == True: print("temperature fields, ", end = "")
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

if temperature:
	metafile.append('temperature.%03.0d-001.meta' % ngroup)
	filelist.append('filelist_temperature')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 temperature-%08.0d.%03.0d-001 >> filelist_temperature' % (i,ngroup))

if colcdsvel==1:
	metafile.append('')
	filelist.append('filelist_colloid')
	for i in range(nstart,nend+nint,nint):
		os.system('ls -t1 config.cds%08.0d.001-001 >> filelist_colloid' % i)

if total_flux_psi:
	metafile.append('total_flux_psi.00%d-001.meta' % ngroup)
	filelist.append('filelist_total_flux_psi')
	for i in range(nstart,nend+nint,nint):
          os.system('ls -t1 total_flux_psi-%08.0d.00%d-001 >> filelist_total_flux_psi' % (i,ngroup))

if advective_flux_psi:
	metafile.append('advective_flux_psi.00%d-001.meta' % ngroup)
	filelist.append('filelist_advective_flux_psi')
	for i in range(nstart,nend+nint,nint):
          os.system('ls -t1 advective_flux_psi-%08.0d.00%d-001 >> filelist_advective_flux_psi' % (i,ngroup))
if mu:
	metafile.append('mu.00%d-001.meta' % ngroup)
	filelist.append('filelist_mu')
	for i in range(nstart,nend+nint,nint):
          os.system('ls -t1 mu-%08.0d.00%d-001 >> filelist_mu' % (i,ngroup))
if fed:
	metafile.append('fed.00%d-001.meta' % ngroup)
	filelist.append('filelist_fed')
	for i in range(nstart,nend+nint,nint):
          os.system('ls -t1 fed-%08.0d.00%d-001 >> filelist_fed' % (i,ngroup))



# Create vtk-files
for i in range(len(filelist)):
	if filelist[i] == 'filelist_vel' or filelist[i] == 'filelist_phi' or filelist[i] == 'filelist_temperature' or filelist[i] == 'filelist_total_flux_psi' or filelist[i] == 'filelist_advective_flux_psi' or filelist[i] == ' filelist_mu' or filelist[i] == 'filelist_fed':

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
