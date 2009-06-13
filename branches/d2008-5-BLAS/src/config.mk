
# determine which machine and apply appropriate compiler settings
# 
###########################################################################
#Define Machine.
#
#MACHINE = HPCX
MACHINE = Ness
#MACHINE = HecToR
#MACHINE = ECDF

# choose 'single' or 'binary' fluid scheme
#SCHEME= single

# chose whether to use got blas
#GOTO = goto
###########################################################################

ifeq ($(MACHINE),HPCX)
	CC=xlc_r
	MPICC=mpcc_r
	OPTS = -D_D3Q19_ 
	CFLAGS=$(OPTS) -q64 -DHPCX
	LIBS= -lessl
else
	ifeq ($(MACHINE),Ness)
		CC=gcc
		MPICC=mpicc 
		OPTS = -D_D3Q19_ -DACML -fastsse
		CFLAGS=$(OPTS) -Minform=warn -Msafeptr -DNDEBUG -Mipa=inline
		LIBS= -lm -lacml -lpgftnrtl -lrt
	else
		ifeq ($(MACHINE),HecToR)
			CC=gcc
			MPICC=cc
			OPTS = -D_D3Q19_ -DACML
			CFLAGS=$(OPTS) -Minform=warn -O3 -DNDEBUG 
			LIBS= -lm -lacml -lpgftnrtl -lrt
		else
			ifeq ($(MACHINE), ECDF)
				CC=gcc
				MPICC=mpicc
				OPTS = -D_D3Q19_ -DMKL
				CFLAGS=$(OPTS) -DNDEBUG -fast
				#LIBS= -L/exports/applications/apps/intel/mkl/10.0.1.014/lib/em64t/ \
				# -lguide -lmkl_em64t  -lm
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

ifeq ($(GOTO),goto)
        ifeq ($(MACHINE),Ness)
                LIBS= -mp -lm -L./ -lgoto
        endif
        ifeq ($(MACHINE),ECDF)
                LIBS= -lm -L./ -lgoto
        endif
        ifeq ($(MACHINE),HPCX)
                LIBS= libgoto_power5-r1.26.a -bmap:map
        endif
endif


ifeq ($(SCHEME), single)
	OPTS += -D_SINGLE_FLUID_
else
	OPTS += -D_BINARY_FLUID_
endif

###########################################################################
