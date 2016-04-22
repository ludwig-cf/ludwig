/*
 * targetDP.h: definitions, macros and declarations for targetDP.
 * Alan Gray
 * 
 * Copyright 2015 The University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _TDP_INCLUDED
#define _TDP_INCLUDED

/* KS. Additions */

#include "target_api.h"

__host__ int target_thread_info(void);
__inline__ __device__ int target_block_reduce_sum_int(int * val);

__inline__ __device__ void target_atomic_add_int(int * sum, int  val);

/*

__host__ int host block_reduce_sum_double(double * sum, double val);
__host__ int host_block_reduce_min_double(double * dmin, double val);
__host__ int host_block_reduce_max_double(double * dmax, double val);

__target__ int target_block_reduce_sum_int(int * isum, int ival)
__target__ int target_block_reduce_sum_double(double * sum, double val);
__target__ int target_block_reduce_min_double(double * dmin, double val);
__target__ int target_block_reduce_max_double(double * dmax, double val);

__target__ int target_atomic_sum_double(double * sum, double val);
__target__ int target_atomic_min_double(double * dmin, double val);
__target__ int target_atomic_max_double(double * dmax, double val);

 */

__host__ __device__ int targetGetDeviceCount(int * device);
__host__ __device__ int targetDeviceSynchronise(void);

/* KS. End additions */


/* Main settings */

#define VVL_CUDA 1 /* virtual vector length for TARGETDP CUDA (usually 1) */
#define VVL_C 1 /* virtual vector length for TARGETDP C (usually 1 for AoS */
              /*  or a small multiple of the hardware vector length for SoA)*/

/* End main settings */


#ifdef __NVCC__ /* CUDA */


/*
 * CUDA Settings 
 */

#define DEFAULT_TPB 128 /* default threads per block */

/* Instruction-level-parallelism vector length  - to be tuned to hardware*/
#define VVL VVL_CUDA


/*
 * Language Extensions 
 */


/* The __targetEntry__ keyword is used in a function declaration or definition
 * to specify that the function should be compiled for the target, and that it will be
 * called directly from host code. */

#define __targetEntry__ __global__

/* The __target__ keyword is used in a function declaration or definition to spec-
 * ify that the function should be compiled for the target, and that it will be called
 * from a targetEntry or another target function. */
#define __target__ __device__ 


/* The __targetHost__ keyword is used in a function declaration or definition to
 * specify that the function should be compiled for the host. */
#define __targetHost__ extern "C" __host__


/* The __targetConst__ keyword is used in a variable or array declaration to
 *  specify that the corresponding data can be treated as constant 
 * (read-only) on the target. */
#define __targetConst__ __constant__

/* The __targetLaunch__ syntax is used to launch a function across 
 * a data parallel target architecture. */
#define __targetLaunch__(extent) \
  <<<((extent/VVL)+DEFAULT_TPB-1)/DEFAULT_TPB,DEFAULT_TPB>>>

/* as above but with stride of 1 */
#define __targetLaunchNoStride__(extent) \
  <<<((extent)+DEFAULT_TPB-1)/DEFAULT_TPB,DEFAULT_TPB>>>
  

/* Thread-level-parallelism execution macro */

/* The __targetTLP__ syntax is used, within a __targetEntry__ function, to
 * specify that the proceeding block of code should be executed in parallel and
 * mapped to thread level parallelism (TLP). Note that he behaviour of this op-
 * eration depends on the defined virtual vector length (VVL), which controls the
 * lower-level Instruction Level Parallelism (ILP)  */
#define __targetTLP__(simtIndex,extent) \
  simtIndex = VVL*(blockIdx.x*blockDim.x+threadIdx.x);	\
  if (simtIndex < extent)

/* as above but with stride of 1 */
#define __targetTLPNoStride__(simtIndex,extent) \
  simtIndex = (blockIdx.x*blockDim.x+threadIdx.x);	\
  if (simtIndex < extent)


/* Instruction-level-parallelism execution macro */
/* The __targetILP__ syntax is used, within a __targetTLP__ region, to specify
 * that the proceeding block of code should be executed in parallel and mapped to
 * instruction level parallelism (ILP), where the extent of the ILP is defined by the
 * virtual vector length (VVL) in the targetDP implementation. */
#if VVL == 1
#define __targetILP__(vecIndex)  vecIndex = 0;
#else
#define __targetILP__(vecIndex)  for (vecIndex = 0; vecIndex < VVL; vecIndex++) 
#endif


/* Functions */

/* The targetConstAddress function provides the target address for a constant
 *  object. */
#define targetConstAddress(addr_of_ptr,const_object) \
  cudaGetSymbolAddress(addr_of_ptr, const_object); \
  checkTargetError("__getTargetConstantAddress__"); 

/* The copyConstToTarget function copies data from the host to the target, 
 * where the data will remain constant (read-only) during the execution of 
 * functions on the target. */
#define copyConstToTarget(data_d, data, size) \
  cudaMemcpyToSymbol(*data_d, (const void*) data, size, 0,cudaMemcpyHostToDevice); \
   checkTargetError("copyConstToTarget"); 

/* The copyConstFromTarget function copies data from a constant data location
 *  on the target to the host. */
#define copyConstFromTarget(data, data_d, size) \
  cudaMemcpyFromSymbol((void*) data, *data_d, size, 0,cudaMemcpyDeviceToHost); \
   checkTargetError("__copyConstantFromTarget__"); 





#else /* C versions of the above*/

/* SEE ABOVE FOR DOCUMENTATION */

/* Settings */

/* Instruction-level-parallelism vector length  - to be tuned to hardware*/
#if ! defined(VVL)
#define VVL VVL_C
#endif

/* Language Extensions */

#define HOST
#define __targetHost__

/* kernel function specifiers */
#define __target__
#define __targetEntry__

/* constant data specifier */
#define __targetConst__ 

/* special kernel launch syntax */
#define __targetLaunch__(extent)
#define __targetLaunchNoStride__(extent)


/* Thread-level-parallelism execution macro */

#ifdef _OPENMP

#define __targetTLP__(simtIndex,extent)	\
_Pragma("omp parallel for")				\
for(simtIndex=0;simtIndex<extent;simtIndex+=VVL)

#define __targetTLPNoStride__(simtIndex,extent)   	\
_Pragma("omp parallel for")				\
for(simtIndex=0;simtIndex<extent;simtIndex++)

#else /* NOT OPENMP */

#define __targetTLP__(simtIndex,extent)	\
for(simtIndex=0;simtIndex<extent;simtIndex+=VVL)

#define __targetTLPNoStride__(simtIndex,extent)   	\
for(simtIndex=0;simtIndex<extent;simtIndex++)

#endif



/* Instruction-level-parallelism execution macro */
/* The __targetILP__ syntax is used, within a __targetTLP__ region, to specify
 * that the proceeding block of code should be executed in parallel and mapped to
 * instruction level parallelism (ILP), where the extent of the ILP is defined by the
 * virtual vector length (VVL) in the targetDP implementation. */
#if VVL == 1
#define __targetILP__(vecIndex) vecIndex = 0;
#else

#ifdef _OPENMP
#define __targetILP__(vecIndex)  \
_Pragma("omp simd")				\
 for (vecIndex = 0; vecIndex < VVL; vecIndex++) 
#else
#define __targetILP__(vecIndex)  \
 for (vecIndex = 0; vecIndex < VVL; vecIndex++) 
#endif

#endif


/* functions */

#define targetConstAddress(addr_of_ptr,const_object) \
  *addr_of_ptr=&(const_object);


#define copyConstToTarget(data_d, data, size) \
  memcpy(data_d,data,size);


#define copyConstFromTarget(data, data_d, size) \
  memcpy(data,data_d,size);


#endif



/* Common */

#define NILP VVL

/* Utility functions for indexing */

#define targetCoords3D(coords,extents,index)					\
  coords[0]=(index)/(extents[1]*extents[2]);				\
  coords[1] = ((index) - extents[1]*extents[2]*coords[0]) / extents[2];	\
  coords[2] = (index) - extents[1]*extents[2]*coords[0]			\
    - extents[2]*coords[1]; 

#define targetIndex3D(coords0,coords1,coords2,extents)	\
  extents[2]*extents[1]*(coords0)				\
  + extents[2]*(coords1)					\
  + (coords2); 

enum {TARGET_HALO,TARGET_EDGE};

/* API */
/* see specification or implementation for documentation on these */
__targetHost__ void targetMalloc(void **address_of_ptr,const size_t size);
__targetHost__ void targetCalloc(void **address_of_ptr,const size_t size);
__targetHost__ void targetMallocUnified(void **address_of_ptr,const size_t size);
__targetHost__ void targetCallocUnified(void **address_of_ptr,const size_t size);
__targetHost__ void copyToTarget(void *targetData,const void* data,size_t size);
__targetHost__ void copyFromTarget(void *data,const void* targetData,size_t size);
__targetHost__ void targetInit3D(int extents[3], size_t nfieldsmax, int nhalo);
__targetHost__ void targetFinalize3D();
__targetHost__ void targetInit(int extents[3], size_t nfieldsmax, int nhalo);
__targetHost__ void targetFinalize();
__targetHost__ void checkTargetError(const char *msg);

__targetHost__ void copyToTargetMasked(double *targetData,const double* data,size_t nsites,
			size_t nfields,char* siteMask);
__targetHost__ void copyFromTargetMasked(double *data,const double* targetData,size_t nsites,
			size_t nfields,char* siteMask);
__targetHost__ void copyToTargetMaskedAoS(double *targetData,const double* data,size_t nsites,
			size_t nfields,char* siteMask);
__targetHost__ void copyFromTargetMaskedAoS(double *data,const double* targetData,size_t nsites,
			size_t nfields,char* siteMask);

//__targetHost__ void copyFromTargetBoundary3D(double *data,const double* targetData,int extents[3], size_t nfields, int offset,int depth);
__targetHost__ void copyFromTarget3DEdge(double *data,const double* targetData,int extents[3], size_t nfields);
__targetHost__ void copyToTarget3DHalo(double *targetData,const double* data, int extents[3], size_t nfields);
__targetHost__ void copyFromTargetPointerMap3D(double *data,const double* targetData, int extents[3], size_t nfields, int includeNeighbours, void** ptrarray);
__targetHost__ void copyToTargetPointerMap3D(double *targetData,const double* data, int extents[3], size_t nfields, int includeNeighbours, void** ptrarray);
__targetHost__ void targetSynchronize();
__targetHost__ void targetFree(void *ptr);
__targetHost__ void checkTargetError(const char *msg);
__targetHost__ void targetFree(void *ptr);
__targetHost__ void targetZero(double* array,size_t size);
//__targetHost__ double targetDoubleSum(double* array, size_t size);
__targetHost__ void targetAoS2SoA(double* array, size_t nsites, size_t nfields);
__targetHost__ void targetSoA2AoS(double* array, size_t nsites, size_t nfields);



__targetHost__ void copyDeepDoubleArrayToTarget(void* targetObjectAddress,void* hostObjectAddress,void* hostComponentAddress,int size);

__targetHost__ void copyDeepDoubleArrayFromTarget(void* hostObjectAddress,void* targetObjectAddress,void* hostComponentAddress,int size);

#endif
