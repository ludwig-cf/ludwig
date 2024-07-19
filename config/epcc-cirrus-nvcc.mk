###############################################################################
#
#  nvcc build
#
#  "Serial" build.
#
#  module load nvidia/nvhpc-nompi
#
###############################################################################

BUILD  = serial
MODEL  = -D_D3Q19_
TARGET = nvcc

CC     = nvcc
CFLAGS = -g -DADDR_SOA -O2 -arch=sm_70 -x cu -dc

# PTX assembler extra information:  -Xptxas -v
# Alternative compiler, e.g., Intel: -ccbin=icpc -Xcompiler -fast

AR = ar
ARFLAGS = -cr
LDFLAGS = -arch=sm_70
