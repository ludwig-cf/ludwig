/*
 * targetDP.h: definitions, macros and declarations for targetDP.
 * Alan Gray, November 2013
 */

#ifndef _DATA_PARALLEL_INCLUDED
#define _DATA_PARALLEL_INCLUDED


/* Language "extensions", implemented through preprocessor */


#ifdef CUDA /* CUDA */

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


#else /* X86 */

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
//#define TARGET_TLP(simtIndex,extent)    _Pragma("omp parallel for")	\
//  for(simtIndex=0;simtIndex<extent;simtIndex+=NILP)

#define TARGET_TLP(simtIndex,extent)   	\
  for(simtIndex=0;simtIndex<extent;simtIndex+=NILP)

#endif


/* Common */

/* Initialisation */
#define TARGET_INDEX_INIT(extent)		\
  int targetExtent=extent;			\
  int baseIndex=0,vecIndex=0;


#define ILPIDX(instrn) (instrn)*NILP+vecIndex 

/* Instruction-level-parallelism execution macro */
#define TARGET_ILP  for (vecIndex = 0; vecIndex < NILP; vecIndex++) 

/* declaration of thread-dependent stack data */
#define VDECLSC(var) var[NILP]
#define VDECL1D(var,extent) var[extent*NILP]
#define VDECL2D(var,extent1,extent2) var[extent1][extent2*NILP]

/* access functions for thread-dependent stack data */
#define VSC(var) var[vecIndex]
#define V1D(var,idx) var[ILPIDX(idx)]
#define V2D(var,idx1,idx2) var[idx1][ILPIDX(idx2)]

/* access function for lattice site data */
#define SITE(array,field) \
  array[targetExtent*p+baseIndex+vecIndex]



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


#endif
