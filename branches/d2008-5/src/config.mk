
# determine which machine and apply appropriate compiler settings
# 
###########################################################################
#Define Machine.
#

MACHINE = HPCX
#MACHINE = Ness
#MACHINE = HecToR
#MACHINE = ECDF

# choose 'single' or 'binary' fluid scheme
#SCHEME= single

###########################################################################

ifeq ($(MACHINE),HPCX)
	CC=xlc_r
	MPICC=mpcc_r
	OPTS = -D_D3Q19_ -DNDEBUG 
	CFLAGS=$(OPTS) -q64
	LIBS= -lm
else
	ifeq ($(MACHINE),Ness)
		CC=gcc
		MPICC=mpicc
		OPTS = -D_D3Q19_
		CFLAGS=$(OPTS) -g -Minform=warn -O3 -DNDEBUG
	else
		ifeq ($(MACHINE),HecToR)
			CC=
			MPICC=cc
			OPTS = -D_D3Q19_
			CFLAGS=$(OPTS) -g -Minform=warn -O3 -DNDEBUG 
			LIBS= -lm
		else
			ifeq ($(MACHINE), ECDF)
				CC=gcc
				MPICC=mpicc 
				OPTS = -D_D3Q19_ -fast
				CFLAGS=$(OPTS) -DNDEBUG 
				LIBS= -lm
			else
				echo	
				echo "OS not defined !!" 
				echo
			endif
		endif
	endif
endif

ifeq ($(SCHEME), single)
	OPTS += -D_SINGLE_FLUID_
else
	OPTS += -D_BINARY_FLUID_
endif

###########################################################################
