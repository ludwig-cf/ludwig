CCS = xlc
CCP = mpixlc

INCS	=	
 
# BE:     "undef" is Doi-Olmsted, "def" is Beris-Edwards
# CHOL:   "undef" is nematic uniaxial, "def" is cholesteric
# EON:    "undef" electric field off, "def" electric field on
# BC:     boundary conditions on/off
# FIXEDQ: fixed or free boundary conditions for Q
 

DEFS    =	 -DBE -DEON -DCHOL \
                 -DMPICH_IGNORE_CXX_SEEK
#-O3 -qarch=440d -qipa\

LIBS	=	-lstdc++ -libmc++ -lsupc++

LIBPATH =

NAME	= HLBLC08

all: $(NAME)S $(NAME)P

String.o: String.cc
	$(CCP) -c String.cc $(DEFS)

LCParallel.o: LCParallel.cc
	$(CCP) -c LCParallel.cc -DPARALLEL

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

# DO NOT DELETE

String.o: String.hh Bool.hh
LCParallel.o: LCParallel.hh /usr/include/math.h /usr/include/features.h
LCParallel.o: /usr/include/sys/cdefs.h /usr/include/gnu/stubs.h
LCParallel.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
LCParallel.o: /usr/include/bits/mathcalls.h /usr/include/stdlib.h
LCParallel.o: /usr/lib/gcc-lib/powerpc-suse-linux/3.3.3/include/stddef.h
LCParallel.o: /usr/include/sys/types.h /usr/include/bits/types.h
LCParallel.o: /usr/include/bits/wordsize.h /usr/include/bits/typesizes.h
LCParallel.o: /usr/include/time.h /usr/include/endian.h
LCParallel.o: /usr/include/bits/endian.h /usr/include/sys/select.h
LCParallel.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
LCParallel.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
LCParallel.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/sched.h
LCParallel.o: /usr/include/alloca.h LCMain_hybrid.hh String.hh Bool.hh
LCMain_h.o: LCMain_hybrid.hh String.hh Bool.hh
