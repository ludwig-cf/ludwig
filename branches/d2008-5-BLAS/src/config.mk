
# determine which machine and apply appropriate compiler settings
# 
###########################################################################
#Define Machine.
#
#MACHINE = HPCX
#MACHINE = Ness
MACHINE = HecToR
#MACHINE = ECDF

# choose compiler suite for Hector Machine option
#PRG_ENV=pgi

# binary or single fluid
#SCHEME= single

# chose whether to use got blas
GOTO = goto
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
		OPTS = -D_D3Q19_ -fastsse
		CFLAGS=$(OPTS) -DNDEBUG -Minform=warn -Msafeptr -Mipa=inline,fast 
		LIBS= -DACML -lm -lacml -lpgftnrtl -lrt
	else
		ifeq ($(MACHINE),HecToR)
			ifeq ($(PRG_ENV), pgi)
				CC=
				MPICC=cc
				OPTS = -D_D3Q19_ -DACML -fastsse
				CFLAGS=$(OPTS) -DNDEBUG -Minform=warn -Msafeptr -Mipa=inline,fast 
				LIBS=   -DACML -lacml -lrt  -lpgftnrtl -lm
			else
				CC=cc
                                MPICC=cc
                                OPTS = -D_D3Q19_ -DNDEBUG # -DACML
                                CFLAGS=$(OPTS) -O3 -OPT:Ofast -OPT:recip=ON -OPT:malloc_algorithm=1 -inline \
				 -INLINE:preempt=ON -march=auto -m64 -msse3 -LNO:simd=2
                                LIBS=   -lacml -lm
			endif
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
                LIBS= -lm -L./ -lgoto
        endif
        ifeq ($(MACHINE),ECDF)
                LIBS= -lm -L./ -lgoto
        endif
        ifeq ($(MACHINE),HPCX)
                LIBS= libgoto_power5-r1.26.a -bmap:map
        endif
        ifeq ($(MACHINE), HecToR)
                LIBS= -lm -L./ -lgoto_barcelonap-r1.26
        endif
endif


ifeq ($(SCHEME), single)
	OPTS += -D_SINGLE_FLUID_
else
	OPTS += -D_BINARY_FLUID_
endif

###########################################################################
