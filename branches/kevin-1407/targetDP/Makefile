# 
# Makefile
# Copyright 2015 The University of Edinburgh
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
# http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.  
#

include ../Makefile.mk

MF= Makefile

SRC_C= targetDP_C.c 
SRC_CUDA= targetDP_CUDA.c 




AR = ar -cru

.SUFFIXES:
.SUFFIXES: .c  .o

CUARCH= sm_35


USEGPU=
ifeq ($(TARGETDP),GPU)
USEGPU=1
endif

ifeq ($(MYHOSTNAME),mbpag3)
CC= gcc-4.9
NVCC= nvcc #-ccbin=gcc-4.9
CUARCH=sm_30
endif



# CFLAGS = -O3 -fopenmp  #-save-temps
# CFLAGS = -g -G #-save-temps
# CFLAGS = -g -fopenmp  #-save-temps

#mic
# CC= icc
# CFLAGS_BASE= -O3 -openmp -mmic -I $(TARGETDPDIR) #-vec-report=2
# LFLAGS=$(CFLAGS_BASE)
# CFLAGS=$(CFLAGS_BASE)
#endif

 CC_CUDA=	$(NVCC)
 CFLAGS_CUDA= -DCUDA -O3 -arch=$(CUARCH) -I $(TARGETDPDIR) -dc -x cu #-save-temps
# CFLAGS_CUDA= -DCUDA -g -G -arch=sm_30 -I $(TARGETDPDIR) -dc -x cu #-save-temps

DEFTAR=targetDP_C

ifdef USEGPU
DEFTAR+= targetDP_CUDA
endif


default: $(DEFTAR)

all: targetDP_CUDA targetDP_C

targetDP_CUDA:
	$(CC_CUDA) $(CFLAGS_CUDA) -c targetDP_CUDA.c
	$(AR) libtargetDP_CUDA.a targetDP_CUDA.o

targetDP_C:
	$(CC) $(CFLAGS) -c targetDP_C.c
	$(AR) libtargetDP_C.a targetDP_C.o

$(OBJ):	$(MF) *.h 

clean:
	rm -f *.o *.a
