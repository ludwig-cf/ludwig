
# apply appropriate compiler settings, may have to use gmake if make fails (stringency issues)
# 
###########################################################################
#choose appropriate HPC Machine.
#
#MACHINE = HPCX
MACHINE = Ness
#MACHINE = HecToR
#MACHINE = ECDF

# binary or single fluid
#SCHEME= single

# to enable blocking
BLOCKING=YES

###########################################################################


ifeq ($(MACHINE),HPCX)
	CC=xlc_r
	MPICC=mpcc_r
	OPTS = -D_D3Q19_ -O5 -qipa
	CFLAGS=$(OPTS) -q64 
	LIBS= -lessl -DPOWER_ESSL -DNDEBUG
else
	ifeq ($(MACHINE),Ness)
		CC=gcc
		MPICC=mpicc 
		OPTS= -D_D3Q19_ -fastsse -Msafeptr -Minfo=all -Mautoinline
		CFLAGS=$(OPTS) -DNDEBUG -Minform=warn -DX86
		LIBS=  -lm -lacml -lpgftnrtl -lrt 

	else
		ifeq ($(MACHINE),HecToR)
			CC=cc	
                        MPICC=cc
                        OPTS = -D_D3Q19_ -DNDEBUG
                        CFLAGS=$(OPTS) -O3 -OPT:Ofast -OPT:recip=ON -OPT:malloc_algorithm=1 -inline \
			-INLINE:preempt=ON -march=auto -m64 -msse3 -LNO:simd=2 -DX86 
                        LIBS=   -lacml -lm

		else
			ifeq ($(MACHINE), ECDF)
				CC=gcc
				MPICC=mpicc
				OPTS = -D_D3Q19_ 
				CFLAGS=$(OPTS) -DNDEBUG -fast -DX86 -axS -funroll-loops -ansi-alias -align -ipo -vec-report
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
                LIBS= libgoto_power5-r1.26.a -bmap:map -DPOWER_GOTO
        endif
        ifeq ($(MACHINE), HecToR)
                LIBS= -lm -L./ -lgoto_barcelonap-r1.26
        endif
endif


ifeq ($(SCHEME), single)
	OPTS += -D_SINGLE_FLUID_
else
	OPTS += -D_BINARY_FLUID_

	ifeq ($(BLOCKING), YES)
		OPTS += -DBLOCKING
	endif
endif

ifeq ($(FUSED), YES)
	OPTS += -D_FUSED_
endif

###########################################################################
