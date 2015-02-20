/*
 * targetDP.h: definitions, macros and declarations for targetDP.
 * Alan Gray, November 2013
 */

#include <stdlib.h>
#include <string.h>

#ifndef _DATA_PARALLEL_INCLUDED
#define _DATA_PARALLEL_INCLUDED


#ifdef CUDA /* CUDA */

/* Settings */
#define DEFAULT_TPB 256 /* default threads per block */

/* Instruction-level-parallelism vector length */
//#define NILP 2
#define NILP 1


/* Language Extensions */

#define HOST extern "C" __host__
#define __targetHost__ extern "C" __host__


/* kernel function specifiers */
#define __target__ __device__ 
#define __targetEntry__ __global__

/* constant data specifier */
#define __targetConst__ __constant__

/* special kernel launch syntax */
#define __targetLaunch__(extent) \
  <<<((extent/NILP)+DEFAULT_TPB-1)/DEFAULT_TPB,DEFAULT_TPB>>>

#define __targetLaunchNoStride__(extent) \
  <<<((extent)+DEFAULT_TPB-1)/DEFAULT_TPB,DEFAULT_TPB>>>
  

/* Thread-level-parallelism execution macro */
#define __targetTLP__(simtIndex,extent) \
  simtIndex = NILP*(blockIdx.x*blockDim.x+threadIdx.x);	\
  if (simtIndex < extent)

#define __targetTLPNoStride__(simtIndex,extent) \
  simtIndex = (blockIdx.x*blockDim.x+threadIdx.x);	\
  if (simtIndex < extent)


/* Functions */

#define targetConstAddress(addr_of_ptr,const_object) \
  cudaGetSymbolAddress(addr_of_ptr, const_object); \
  checkTargetError("__getTargetConstantAddress__"); 

#define copyConstToTarget(data_d, data, size) \
  cudaMemcpyToSymbol(*data_d, (const void*) data, size, 0,cudaMemcpyHostToDevice); \
   checkTargetError("copyConstToTarget"); 

#define copyConstFromTarget(data, data_d, size) \
  cudaMemcpyFromSymbol((void*) data, *data_d, size, 0,cudaMemcpyDeviceToHost); \
   checkTargetError("__copyConstantFromTarget__"); 





#else /* X86 */

/* Settings */

/* Instruction-level-parallelism vector length */
#define NILP 1


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
/* #define __targetTLP__(simtIndex,extent)    _Pragma("omp parallel for")	\
   for(simtIndex=0;simtIndex<extent;simtIndex+=NILP)*/

#define __targetTLP__(simtIndex,extent)   	\
  for(simtIndex=0;simtIndex<extent;simtIndex+=NILP)

#define __targetTLPNoStride__(simtIndex,extent)   	\
  for(simtIndex=0;simtIndex<extent;simtIndex++)


/* functions */

#define targetConstAddress(addr_of_ptr,const_object) \
  *addr_of_ptr=&(const_object);


#define copyConstToTarget(data_d, data, size) \
  memcpy(data_d,data,size);


#define copyConstFromTarget(data, data_d, size) \
  memcpy(data,data_d,size);


#endif



/* Common */

/* depricated? */

/* Initialisation */
/* #define TARGET_INDEX_INIT(extent)		\
  int targetExtent=extent;			\
  int baseIndex=0,vecIndex=0;*/


#define ILPIDX(instrn) (instrn)*NILP+vecIndex 

/* Instruction-level-parallelism execution macro */
#define TARGET_ILP(vecIndex)  for (vecIndex = 0; vecIndex < NILP; vecIndex++) 

/* declaration of thread-dependent stack data */

#define DECLARE_SIMD_SCALAR(type, name) type name[NILP];

#define DECLARE_SIMD_VECTOR1D(type, name, extent) type name[extent*NILP];

#define DECLARE_SIMD_VECTOR2D(type, name, extent1, extent2) \
  type name[extent1][extent2*NILP];


/* access functions for thread-dependent stack data */
#define SIMD_SC_ELMNT(var,vecIndex) var[vecIndex]
#define SIMD_1D_ELMNT(var,idx,vecIndex) var[(idx)*NILP+vecIndex]
#define SIMD_2D_ELMNT(var,idx1,idx2,vecIndex) var[idx1][(idx2)*NILP+vecIndex]

/* access function for lattice site data */
#define SITE(array,field) \
  array[targetExtent*p+baseIndex+vecIndex]

/* end depricated? */


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

void targetInit(size_t nsites,size_t nfieldsmax);
void targetFinalize();
void checkTargetError(const char *msg);
void copyToTarget(void *targetData,const void* data,size_t size);
void copyFromTarget(void *data,const void* targetData,size_t size);
void copyToTargetMasked(double *targetData,const double* data,size_t nsites,
			size_t nfields,char* siteMask);
void copyFromTargetMasked(double *data,const double* targetData,size_t nsites,
			size_t nfields,char* siteMask);
void copyToTargetMaskedAoS(double *targetData,const double* data,size_t nsites,
			size_t nfields,char* siteMask);
void copyFromTargetMaskedAoS(double *data,const double* targetData,size_t nsites,
			size_t nfields,char* siteMask);

void copyFromTargetBoundary3D(double *data,const double* targetData,int extents[3], size_t nfields, int offset,int depth);
void copyToTargetBoundary3D(double *targetData,const double* data,int extents[3], size_t nfields, int offset,int depth);
void targetSynchronize();
void targetFree(void *ptr);
void checkTargetError(const char *msg);
void targetMalloc(void **address_of_ptr,const size_t size);
void targetCalloc(void **address_of_ptr,const size_t size);
void targetFree(void *ptr);

#endif
