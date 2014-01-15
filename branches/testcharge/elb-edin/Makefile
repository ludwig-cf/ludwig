SVNDEF := -DSVN_REV='"\"$(shell svnversion -n .)\""'

CXXFLAGS = -Wall -O3 -DNUM -DFLUID_TYPE=2 -DDIM3=1 $(SVNDEF)
#CXXFLAGS = -g -Wall -DFLUID_TYPE=2 -DDIM3=1 $(SVNDEF) 
LIBPATH = 
LDFLAGS =  -lm   #-pg for profiling 
OBJSEQ = elb.o simpar.o lb.o init.o ch_dynamics.o io.o mc_equilibration.o
#OBJSEQ = lb_charge_comments.o simpar.o lattice_lb.o
INC = *.h

C = gcc
CC = gcc

# Link rule

elb:	$(OBJSEQ)
	$(C) $(LDFLAGS) $(CCFLAGS) $(OBJSEQ) -o $@

clean:
	rm -f -r *.o elb oelb*

# Compilation rules

.c.o:
	$(CC) $(CXXFLAGS)  -c $<

.C.o:
	$(CC) $(CXXFLAGS)  -c $<

.cpp.o:
	$(CC) $(CXXFLAGS)  -c $<


# Individual dependencies

$(OBJSEQ):	$(INC)
