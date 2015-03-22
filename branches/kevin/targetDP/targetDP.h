/*
 * targetDP.h: definitions, macros and declarations for targetDP.
 * Alan Gray, November 2013
 */

#include <stdlib.h>

#ifndef _DATA_PARALLEL_INCLUDED
#define _DATA_PARALLEL_INCLUDED


/* Language "extensions", implemented through preprocessor */

#include "target_api.h"

#ifdef CUDA /* CUDA */

#define HOST extern "C" __host__


/* default threads per block */
#define DEFAULT_TPB 256
//#define DEFAULT_TPB 32

/* kernel function specifiers */
#define TARGET __device__ 
#define TARGET_ENTRY __global__

/* constant data specifier */
#define TARGET_CONST __constant__

/* Instruction-level-parallelism vector length */
//#define NILP 2
#define NILP 1


/* special kernel launch syntax */
#define TARGET_LAUNCH(extent) \
  <<<((extent/NILP)+DEFAULT_TPB-1)/DEFAULT_TPB,DEFAULT_TPB>>>
  

/* Thread-level-parallelism execution macro */
#define TARGET_TLP(simtIndex,extent) \
  simtIndex = NILP*(blockIdx.x*blockDim.x+threadIdx.x);	\
  if (simtIndex < extent)

/* kevin */
#define __target__ TARGET
#define __target_entry__ TARGET_ENTRY
#define target_is_host() 0
#define target_launch(kernel_function, nblocks, ntpb, ...) \
  kernel_function<<<nblocks, ntpb>>>(__VA_ARGS__)

#else /* X86 */

#define HOST
#define __host__
#define __device__
#define __target__
#define __target_entry__
#define __constant__ const

#define target_launch(kernel, nblocks, ntpb, ...) kernel(__VA_ARGS__)
#define target_is_host() 1

/* kernel function specifiers */
#define TARGET 
#define TARGET_ENTRY

/* constant data specifier */
#define TARGET_CONST 

/* special kernel launch syntax */
#define TARGET_LAUNCH(extent)

/* Instruction-level-parallelism vector length */
#define NILP 1

/* Thread-level-parallelism execution macro */
/* #define TARGET_TLP(simtIndex,extent)    _Pragma("omp parallel for")	\
   for(simtIndex=0;simtIndex<extent;simtIndex+=NILP)*/

#define TARGET_TLP(simtIndex,extent)   	\
  for(simtIndex=0;simtIndex<extent;simtIndex+=NILP)

#endif


/* Common */

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


#define GET_3DCOORDS_FROM_INDEX(index,coords,extents)	     \
    coords[0]=index/(extents[1]*extents[2]); \
    coords[1] = (index - extents[1]*extents[2]*coords[0]) / extents[2]; \
    coords[2] = index - extents[1]*extents[2]*coords[0] \
      - extents[2]*coords[1]; 

#define INDEX_FROM_3DCOORDS(coords0,coords1,coords2,extents)	\
  extents[2]*extents[1]*(coords0)				\
  + extents[2]*(coords1)					\
  + (coords2); 


/* API */

typedef double (*mu_fntype)(const int, const int, const double*, const double*);
typedef void (*pth_fntype)(const int, double(*)[3*NILP], const double*, const double*, const double*);


void targetInit(size_t nsites, size_t nfieldsmax);
void targetFinalize();
void checkTargetError(const char *msg);
void targetMalloc(void **address_of_ptr,size_t size);
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
void syncTarget();
void targetFree(void *ptr);
void checkTargetError(const char *msg);
void targetMalloc(void **address_of_ptr,const size_t size);
void targetCalloc(void **address_of_ptr,const size_t size);
void targetFree(void *ptr);
void copyConstantIntToTarget(int *data_d, const int *data, const int size);
void copyConstantInt1DArrayToTarget(int *data_d, const int *data, const int size);
void copyConstantInt2DArrayToTarget(int **data_d, const int *data, const int size);
void copyConstantDoubleToTarget(double *data_d, const double *data, const int size);
void copyConstantDouble1DArrayToTarget(double *data_d, const double *data, const int size);
void copyConstantDouble2DArrayToTarget(double **data_d, const double *data, const int size);
void copyConstantDouble3DArrayToTarget(double ***data_d, const double *data, const int size);

void  copyConstantMufnFromTarget(mu_fntype* data, mu_fntype* data_d, const int size );
void  copyConstantPthfnFromTarget(pth_fntype* data, pth_fntype* data_d, const int size );

void copyConstantDoubleFromTarget(double *data, const double *data_d, const int size);

#endif
