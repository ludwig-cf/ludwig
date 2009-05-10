
# determine which machine and apply appropriate compiler settings
# 
###########################################################################
#Define Machine.
#
#MACHINE = HPCX
#MACHINE = Ness
#MACHINE = HecToR
MACHINE = ECDF
###########################################################################

ifeq ($(MACHINE),HPCX)
	CC=xlc_r
	MPICC=mpcc_r
	OPTS = -D_D3Q19_ 
	CFLAGS=$(OPTS) -q32
	LDFLAGS=
else
	ifeq ($(MACHINE),Ness)
		CC=gcc
		MPICC=mpicc 
		OPTS = -D_D3Q19_ -DACML 
		CFLAGS=$(OPTS) -g -Minform=warn -O3 -DNDEBUG -D_SINGLE_FLUID_ 
		LIBS= -lm -lacml -lpgftnrtl -lrt
	else
		ifeq ($(MACHINE),HecToR)
			CC=gcc
			MPICC=mpicc
			OPTS = -D_D3Q19_
			CFLAGS=$(OPTS) -g -Minform=warn -O3 -DNDEBUG -D_SINGLE_FLUID_
		else
			ifeq ($(MACHINE), ECDF)
				CC=gcc
				MPICC=mpicc
				OPTS = -D_D3Q19_ -DMKL
				CFLAGS=$(OPTS) -DNDEBUG -D_SINGLE_FLUID_ 
				LIBS= -L/exports/applications/apps/intel/mkl/10.0.1.014/lib/em64t/ \
				 -lmkl_intel_lp64  -lmkl_sequential -lmkl_core  -lm
			else
				echo	
				echo "OS not defined !!" 
				echo
			endif
		endif
	endif
endif

###########################################################################
