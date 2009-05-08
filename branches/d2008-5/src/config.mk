
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
	OPTS = -D_D3Q19_ #-DACML
	CFLAGS=$(OPTS) -q32
	LDFLAGS=
else
	ifeq ($(MACHINE),Ness)
		CC=gcc
		MPICC=mpicc
		OPTS = -D_D3Q19_
		CFLAGS=$(OPTS) -g -Minform=warn -O3 -DNDEBUG -D_SINGLE_FLUID_
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
				OPTS = -D_D3Q19_ 
				CFLAGS=$(OPTS) -DNDEBUG -D_SINGLE_FLUID_
			else
				echo	
				echo "OS not defined !!" 
				echo
			endif
		endif
	endif
endif

###########################################################################
