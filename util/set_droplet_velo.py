import sys, os, re, math

# distribution filename
input_filename  = 'dist-00010000.001-001.bak'
output_filename = 'dist-00010000.001-001'

# droplet density and velocity
rho = 1.0
u = [0.01,0,0]

# droplet positions
r1 = [16,16,16]

# droplet diameter
d = 8.0

# box size
Lx = 32
Ly = 32
Lz = 32

r = [r1]

NVEL = 19

# definition of lattice vectors and weights
cv = [[0,  0,  0],
[ 1,  1,  0], [ 1,  0,  1], [ 1,  0,  0],
[ 1,  0, -1], [ 1, -1,  0], [ 0,  1,  1],
[ 0,  1,  0], [ 0,  1, -1], [ 0,  0,  1],
[ 0,  0, -1], [ 0, -1,  1], [ 0, -1,  0],
[ 0, -1, -1], [-1,  1,  0], [-1,  0,  1],
[-1,  0,  0],[ -1,  0, -1], [-1, -1,  0]]

q_ = [
  [[-1.0/3.0, 0.0, 0.0],[ 0.0,-1.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]],
  [[ 2.0/3.0, 1.0, 0.0],[ 1.0, 2.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]],
  [[ 2.0/3.0, 0.0, 1.0],[ 0.0,-1.0/3.0, 0.0],[ 1.0, 0.0, 2.0/3.0]],
  [[ 2.0/3.0, 0.0, 0.0],[ 0.0,-1.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]],
  [[ 2.0/3.0, 0.0,-1.0],[ 0.0,-1.0/3.0, 0.0],[-1.0, 0.0, 2.0/3.0]],
  [[ 2.0/3.0,-1.0, 0.0],[-1.0, 2.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]],
  [[-1.0/3.0, 0.0, 0.0],[ 0.0, 2.0/3.0, 1.0],[ 0.0, 1.0, 2.0/3.0]],
  [[-1.0/3.0, 0.0, 0.0],[ 0.0, 2.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]],
  [[-1.0/3.0, 0.0, 0.0],[ 0.0, 2.0/3.0,-1.0],[ 0.0,-1.0, 2.0/3.0]],
  [[-1.0/3.0, 0.0, 0.0],[ 0.0,-1.0/3.0, 0.0],[ 0.0, 0.0, 2.0/3.0]],
  [[-1.0/3.0, 0.0, 0.0],[ 0.0,-1.0/3.0, 0.0],[ 0.0, 0.0, 2.0/3.0]],
  [[-1.0/3.0, 0.0, 0.0],[ 0.0, 2.0/3.0,-1.0],[ 0.0,-1.0, 2.0/3.0]],
  [[-1.0/3.0, 0.0, 0.0],[ 0.0, 2.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]],
  [[-1.0/3.0, 0.0, 0.0],[ 0.0, 2.0/3.0, 1.0],[ 0.0, 1.0, 2.0/3.0]],
  [[ 2.0/3.0,-1.0, 0.0],[-1.0, 2.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]],
  [[ 2.0/3.0, 0.0,-1.0],[ 0.0,-1.0/3.0, 0.0],[-1.0, 0.0, 2.0/3.0]],
  [[ 2.0/3.0, 0.0, 0.0],[ 0.0,-1.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]],
  [[ 2.0/3.0, 0.0, 1.0],[ 0.0,-1.0/3.0, 0.0],[ 1.0, 0.0, 2.0/3.0]],
  [[ 2.0/3.0, 1.0, 0.0],[ 1.0, 2.0/3.0, 0.0],[ 0.0, 0.0,-1.0/3.0]]];


w0 = 12.0/36.0
w1 = 2.0/36.0
w2 = 1.0/36.0

wv = [w0, 
      w2, w2, w1, w2, w2, w2, 
      w1, w2, w1, w1, w2, w1, 
      w2, w2, w2, w1, w2, w2]

rcs2 = 3.0

def feq(rho, u):

    fp = []
    for p in range(0,NVEL):

	udotc = 0.0
	sdotq = 0.0

	for ia in range(0,3):
	    udotc = udotc + u[ia]*cv[p][ia]
	    for ib in range(0,3):
		sdotq += q_[p][ia][ib]*u[ia]*u[ib];

	fp.append(rho*wv[p]*(1.0 + rcs2*udotc + 0.5*rcs2*rcs2*sdotq));
    return fp

# read and modify distribution file

inp = open(input_filename,'r')
out = open(output_filename,'w')

while 1:
    line=inp.readline()
    if not line: break
    arg = line.split();

    cds = [int(arg[0]), int(arg[1]), int(arg[2])]

    dcdsr = math.sqrt(pow(cds[0]-r1[0],2)+pow(cds[1]-r1[1],2)+pow(cds[2]-r1[2],2)) 

    f = [float(arg[3]),\
	  float(arg[4]), float(arg[5]), float(arg[6]),\
	  float(arg[7]), float(arg[8]), float(arg[9]),\
	  float(arg[10]), float(arg[11]), float(arg[12]),\
	  float(arg[13]), float(arg[14]), float(arg[15]),\
	  float(arg[16]), float(arg[17]), float(arg[18]),\
	  float(arg[19]), float(arg[20]), float(arg[21])]

    for ia in range(0,3):
	out.write('%d ' % cds[ia])

    if dcdsr < d:
	fdrop = []
	fdrop = feq(rho,u)
	for p in range(0,NVEL):
	    out.write('%12.6le ' % fdrop[p])
    else:
	for p in range(0,NVEL):
	    out.write('%12.6le ' % f[p])

    out.write('\n')



