CCS = xlc 
CCP = mpixlc

# issue the following on hector at the command line (redirects compiler)
# module swap PrgEnv-pgi PrgEnv-pathscale


# issue the following on hector at the command line (redirects compiler)
# module swap PrgEnv-pgi PrgEnv-pathscale

INCS	=	
 
# BE:     "undef" is Doi-Olmsted, "def" is Beris-Edwards
# CHOL:   "undef" is nematic uniaxial, "def" is cholesteric
# EON:    "undef" electric field off, "def" electric field on
# BC:     boundary conditions on/off
# FIXEDQ: fixed or free boundary conditions for Q
 

DEFS    =	 -DBE -DEON -DCHOL\
                 -DMPICH_IGNORE_CXX_SEEK -D_COMM_3D_\
		 -D_BINARY_IO_


LIBS	=        -lstdc++ -libmc++ -lsupc++

LIBPATH =

NAME	= HLBLC08

all: $(NAME)P

String.o: String.cc
	$(CCP) -c String.cc $(DEFS)

LCParallel.o: LCParallel.cc
	$(CCP) -c LCParallel.cc -DPARALLEL -D_COMM_3D_

$(NAME)S: String.o Initialization.cc Output.cc LCMain_hybrid.cc
	$(CCS) LCMain_hybrid.cc String.o -o $(NAME)S $(INCS) $(DEFS) \
	$(LIBPATH) $(LIBS)

$(NAME)P: String.o Initialization.cc Output.cc LCParallel.o LCMain_hybrid.cc
	$(CCP) LCMain_hybrid.cc String.o LCParallel.o -o $(NAME)P $(INCS) \
	$(DEFS) -DPARALLEL $(LIBPATH) $(LIBS)

clean:
	rm -f *.o $(NAME)S $(NAME)P

out-clean:
	rm -f *.dat *.log *.err *.out

depend:
	makedepend $(INCS) $(DEFS) $(NAME).cc String.cc LCParallel.cc LCMain_h.cc

