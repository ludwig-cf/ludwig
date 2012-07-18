# controlparamters.py
#
# Script for calculation of cholesteric LC input parameters

import sys, os, re, math

try:
	tau=float(sys.argv[1])
	kappa=float(sys.argv[2])
	e=float(sys.argv[3])
	L1_0=float(sys.argv[4])
	L_uc=float(sys.argv[5])
	Lz=float(sys.argv[6])

except:
	print
	print "usage: python controlparameters.py tau kappa e L1_0 L_uc Lz"
	print
	print "* tau: temperature"
	print "* kappa: chirality"
	print "* e: effective field strength"
	print "* L1_0: elastic constant (use 0.02 as default)"
	print "* L_uc: dimension of unit cell (1/2 pitch length)"
	print "* Lz: z-dimension of actual simulation box (for voltage input)"
	print
	sys.exit(1)

# dielectric anisotropy
epsa=41.4

# functions 

def gamma(tau):
	return 27.0/4.0/(0.25*tau+2.25)

def A0(tau,kappa):
	return 16.0*(L1_0*q0*q0)*(0.25*tau+2.25)/kappa/kappa

def deltaVz(tau,kappa,e):
	return e*(Lz-1.0)*8.0/9.0*math.sqrt(math.pi*A0(tau,kappa)*gamma(tau)*3.0/2.0/epsa)

# pitch = Pi over unit cell size
q0=math.pi/L_uc

gam=gamma(tau)
Abulk=A0(tau,kappa)
delVz=deltaVz(tau,kappa,e)

print 'A0 = %.12f ' % Abulk
print 'gamma = %.12f ' % gam
print 'delVz = %.12f ' % delVz
print ('L1 = %g (BP1 - rescaled to retain chirality); = %g (BP2)' % (0.5*L1_0,L1_0))
print ('pitch = %.12f (BP1); = %.12f (BP2)' % (math.sqrt(2.)*q0, q0))

