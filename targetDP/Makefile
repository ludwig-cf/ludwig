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

AR = ar -cru

.SUFFIXES:
.SUFFIXES: .c  .o


TARGETSRC = targetDP_C.c cuda_stub_api.c

ifeq ($(CC),nvcc)
TARGETSRC = targetDP_CUDA.c
endif

TARGETOBJ = ${TARGETSRC:.c=.o}

default:
	$(CC) $(CFLAGS) -c $(TARGETSRC)
	$(AR) libtarget.a $(TARGETOBJ)

clean:
	rm -f *.o *.a
