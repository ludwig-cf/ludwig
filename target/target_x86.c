/*****************************************************************************
 *
 *  target_x86.c
 *
 *  Implementation is serial or OpenMP.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (Late of this parish)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "target.h"

/* Globally reserved names. */

dim3 threadIdx;
dim3 blockIdx;
dim3 gridDim = {1, 1, 1};
dim3 blockDim = {1, 1, 1};

static tdpError_t lastError = tdpSuccess;
static char lastErrorString[BUFSIZ] = "";
static int staticStream;

/* Utilities */

static void error_boke(int line, tdpError_t error) {

  fprintf(stderr, "File %s line %d error %s\n", __FILE__, line,
	  tdpGetErrorName(error));
  exit(0);
}

#define errors_make_me_boke(error) error_boke(__LINE__, error)

#define error_return_if(expr, error) \
  do { if ((expr)) { 		     \
      lastError = error;	     \
      errors_make_me_boke(error);    \
      return error;		     \
    }				     \
  } while(0)

#define error_return(error) \
  error_return_if(1, error)

void tdpErrorHandler(tdpError_t ifail, const char * file, int line, int fatal) {

  if (ifail != tdpSuccess) {
    printf("tdpErrorHandler: %s:%d %s %s\n", file, line, tdpGetErrorName(ifail),
	   tdpGetErrorString(ifail));
    if (fatal) exit(ifail);
  }

  return;
}

/*****************************************************************************
 *
 *  tdpThreadModelInfo
 *
 *  Provide spme information on the thread model.
 *
 *****************************************************************************/

__host__ tdpError_t tdpThreadModelInfo(FILE * fp) {

  assert(fp);

#ifndef _OPENMP
  fprintf(fp, "Target thread model: None.\n");
#else
  fprintf(fp, "Target thread model: OpenMP.\n");
  fprintf(fp, "OpenMP threads: %d; maximum number of threads: %d.\n",
	  omp_get_max_threads(), omp_get_num_procs());
#endif

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdp_x86_prelaunch
 *
 *  Injected immediately before "kernel launch".
 *
 *****************************************************************************/

__host__ void tdp_x86_prelaunch(dim3 nblocks, dim3 nthreads) {

  gridDim = nblocks;
  blockDim = nthreads;

  /* sanity checks on user settings here... */

  gridDim.x = 1; /* Assert this for host implementation */

  /* In case we request fewer threads than are available: */

  omp_set_num_threads(blockDim.x*blockDim.y*blockDim.z);

  /* Check blockDim, blockIdx ? */
  threadIdx.x = omp_get_thread_num();
  threadIdx.y = 1;
  threadIdx.z = 1;

  return;
}

void tdp_x86_postlaunch(void) {

  /* Reset the default number of threads. */

  omp_set_num_threads(omp_get_max_threads());

  return;
}

/*****************************************************************************
 *
 *  tdpDeviceGetCacheConfig
 *
 *****************************************************************************/

tdpError_t tdpDeviceGetCacheConfig(tdpFuncCache * cacheConfig) {

  *cacheConfig = tdpFuncCachePreferNone;
  return tdpSuccess;
}


/*****************************************************************************
 *
 *  tdpDeviceSetCacheConfig
 *
 *****************************************************************************/

tdpError_t tdpDeviceSetCacheConfig(tdpFuncCache cacheConfig) {

  /* No op. */
  return tdpSuccess;
}



/*****************************************************************************
 *
 *  tdpDeviceSynchronize
 *
 *****************************************************************************/

tdpError_t tdpDeviceSynchronize(void) {

  /* do nothing */

  lastError = tdpSuccess;

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpFree
 *
 *****************************************************************************/

tdpError_t tdpFree(void * devPtr) {

  error_return_if(devPtr == NULL, tdpErrorInvalidDevicePointer);

  free(devPtr);

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpFreeHost
 *
 *****************************************************************************/

tdpError_t tdpFreeHost(void * ptr) {

  free(ptr);

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpDeviceGetAttribute
 *
 *****************************************************************************/

tdpError_t tdpDeviceGetAttribute(int * value, tdpDeviceAttr attr, int device) {

  assert(value);
  assert(0); /* Return some useful information please */

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  Return id of device currently being used.
 *
 *****************************************************************************/

tdpError_t tdpGetDevice(int * device) {

  assert(device);

  *device = 0;

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpGetDeviceCount
 *
 *  Return number of available devices
 *
 *****************************************************************************/

tdpError_t tdpGetDeviceCount(int * device) {

  *device = 0;

#ifdef FAKE_DEVICE /* "Fake" device */
  *device = 1;
#endif

  /* Strictly, we should return tdpErrorInsufficientDriver or ... */

  return tdpErrorNoDevice;
}

/*****************************************************************************
 *
 *  tdpGetDeviceProperties
 *
 *****************************************************************************/

tdpError_t tdpGetDeviceProperties(struct tdpDeviceProp * prop, int device) {

  prop->maxThreadsPerBlock = TARGET_MAX_THREADS_PER_BLOCK;
  prop->maxThreadsDim[0]   = TARGET_MAX_THREADS_PER_BLOCK;
  prop->maxThreadsDim[1]   = 1;
  prop->maxThreadsDim[2]   = 1;

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpSetDevice
 *
 *****************************************************************************/

tdpError_t tdpSetDevice(int device) {

  error_return_if(device < 0, tdpErrorInvalidDevice);
  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpGetErrorName
 *
 *****************************************************************************/

#define CASE_RETURN(x) case(x): return #x; break

const char *  tdpGetErrorName(tdpError_t error) {

  switch (error) {
    CASE_RETURN(tdpSuccess);
    CASE_RETURN(tdpErrorMissingConfiguration);
    CASE_RETURN(tdpErrorMemoryAllocation);
    CASE_RETURN(tdpErrorInitializationError);
    CASE_RETURN(tdpErrorLaunchFailure);
    CASE_RETURN(tdpErrorLaunchTimeout);
    CASE_RETURN(tdpErrorLaunchOutOfResources);
    CASE_RETURN(tdpErrorInvalidDeviceFunction);
    CASE_RETURN(tdpErrorInvalidSymbol);
    CASE_RETURN(tdpErrorInvalidDevicePointer);
    CASE_RETURN(tdpErrorInvalidResourceHandle);
  default:
    fprintf(stderr, "Unrecognised error code was %d\n", error);
  }

  return "Unrecognised error code";
}

/*****************************************************************************
 *
 *  tdpGetErrorString
 *
 *****************************************************************************/

const char * tdpGetErrorString(tdpError_t ifail) {

  return "";
}

/*****************************************************************************
 *
 *  tdpPeekAtLastError
 *
 *****************************************************************************/

tdpError_t tdpPeekAtLastError(void) {

  return lastError;
}

/*****************************************************************************
 *
 *  tdpGetLastError
 *
 *****************************************************************************/

tdpError_t tdpGetLastError(void) {

  tdpError_t last = lastError;

  lastError = tdpSuccess;
  strcpy(lastErrorString, "");

  return last;
}

/*****************************************************************************
 *
 *  tdpGetSymbolAddress
 *
 *****************************************************************************/

tdpError_t tdpGetSymbolAddress(void ** devptr, const void * symbol) {

  assert(devptr);
  assert(symbol);

  error_return_if(symbol == NULL, tdpErrorInvalidSymbol);

  *devptr = (void *) symbol;

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpHostAlloc
 *
 *****************************************************************************/

tdpError_t tdpHostAlloc(void ** phost, size_t size, unsigned int flags) {

  void * ptr = NULL;

  error_return_if(phost == NULL, tdpErrorInvalidValue);

  switch (flags) {
  case tdpHostAllocDefault:
  case tdpHostAllocPortable:
  case tdpHostAllocMapped:
  case tdpHostAllocWriteCombined:

    ptr = malloc(size);
    error_return_if(ptr == NULL, tdpErrorMemoryAllocation);

    *phost = ptr;
    break;

  default:
    error_return(tdpErrorInvalidValue);
  }

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpMalloc
 *
 *****************************************************************************/

tdpError_t tdpMalloc(void ** devPtr, size_t size) {

  assert(devPtr);

  *devPtr = malloc(size);

  error_return_if(*devPtr == NULL, tdpErrorMemoryAllocation);

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpMallocManaged
 *
 *****************************************************************************/

tdpError_t tdpMallocManaged(void ** devptr, size_t size, unsigned int flag) {

  void * ptr = NULL;
  unsigned int valid = (tdpMemAttachGlobal | tdpMemAttachHost);

  assert(devptr);

  error_return_if(size < 1, tdpErrorInvalidValue);
  error_return_if((flag & (~valid)), tdpErrorInvalidValue);

  ptr = malloc(size);
  error_return_if(ptr == NULL, tdpErrorMemoryAllocation);

  *devptr = ptr;

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpMemcpy
 *
 *****************************************************************************/

tdpError_t tdpMemcpy(void * dst, const void * src, size_t count,
		       tdpMemcpyKind kind) {

  assert(dst);
  assert(src);

  error_return_if(count < 1, tdpErrorInvalidValue);

  switch (kind) {
  case tdpMemcpyHostToDevice:
    error_return_if(dst == NULL, tdpErrorInvalidDevicePointer);
    memcpy(dst, src, count);
    break;
  case tdpMemcpyDeviceToHost:
    error_return_if(src == NULL, tdpErrorInvalidDevicePointer);
    memcpy(dst, src, count);
    break;
  case tdpMemcpyHostToHost:
    memcpy(dst, src, count);
    break;
  case tdpMemcpyDeviceToDevice:
    memcpy(dst, src, count);
    break;
  case tdpMemcpyDefault:
  default:
    error_return(tdpErrorInvalidMemcpyDirection);
  }

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpMemcpyFromSymbol
 *
 *****************************************************************************/

tdpError_t tdpMemcpyFromSymbol(void * dst, const void * symbol,
			       size_t count, size_t offset,
			       tdpMemcpyKind kind) {
  assert(dst);
  assert(symbol);

  error_return_if(count < 1, tdpErrorInvalidValue);
  error_return_if(offset != 0, tdpErrorInvalidValue);

  switch (kind) {
  case tdpMemcpyDefault:
  case tdpMemcpyDeviceToHost:
    error_return_if(symbol == NULL, tdpErrorInvalidSymbol);
    memcpy(dst, symbol, count);
    break;
  case tdpMemcpyDeviceToDevice:
    error_return_if(dst == NULL, tdpErrorInvalidDevicePointer);
    error_return_if(symbol == NULL, tdpErrorInvalidSymbol);
    memcpy(dst, symbol, count);
    break;
  case tdpMemcpyHostToDevice:
    assert(0);
  case tdpMemcpyHostToHost:
    assert(0);
  default:
    error_return(tdpErrorInvalidMemcpyDirection);
  }

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpMemcpyToSymbol
 *
 *  CUDA  wants "const void * symbol", but this is avoided as we need
 *  a memset(void * dst, const void * src, ...) .
 *
 *****************************************************************************/

tdpError_t tdpMemcpyToSymbol(void * symbol, const void * src,
			     size_t count, size_t offset,
			     tdpMemcpyKind kind) {
  assert(symbol);
  assert(src);

  error_return_if(count < 1, tdpErrorInvalidValue);
  error_return_if(offset != 0, tdpErrorInvalidValue);

  switch (kind) {
  case tdpMemcpyDefault:
  case tdpMemcpyHostToDevice:
    error_return_if(symbol == NULL, tdpErrorInvalidSymbol);
    memcpy(symbol, src, count);
    break;
  case tdpMemcpyDeviceToDevice:
    error_return_if(src == NULL, tdpErrorInvalidDevicePointer);
    memcpy(symbol, src, count);
    break;
  case tdpMemcpyDeviceToHost:
  case tdpMemcpyHostToHost:
  default:
    error_return(tdpErrorInvalidMemcpyDirection);
  }

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpMemset
 *
 *****************************************************************************/

tdpError_t tdpMemset(void * devPtr, int value, size_t count) {

  error_return_if(devPtr == NULL, tdpErrorInvalidDevicePointer);
  error_return_if(value < 0, tdpErrorInvalidValue);
  error_return_if(value > 255, tdpErrorInvalidValue);

  memset(devPtr, value, count);

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpStreamCreate
 *
 *****************************************************************************/

tdpError_t tdpStreamCreate(tdpStream_t * stream) {

  error_return_if(stream == NULL, tdpErrorInvalidValue);

  *stream = &staticStream;

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpStreamDestroy
 *
 *****************************************************************************/

tdpError_t tdpStreamDestroy(tdpStream_t stream) {

  error_return_if(stream != &staticStream, tdpErrorInvalidResourceHandle);

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpStreamDestroy
 *
 *****************************************************************************/

tdpError_t tdpStreamSynchronize(tdpStream_t stream) {

  error_return_if(stream != &staticStream, tdpErrorInvalidResourceHandle);

  /* Success */

  return tdpSuccess;
}

/*****************************************************************************
 *
 *  tdpMemcpyAsync
 *
 *****************************************************************************/

tdpError_t tdpMemcpyAsync(void * dst, const void * src, size_t count,
			  tdpMemcpyKind kind, tdpStream_t stream) {

  /* Just ignore the stream argument and copy immediately */

  return tdpMemcpy(dst, src, count, kind);
}

static int int_max(int a, int b) {return (a > b) ?a :b;}
static int int_min(int a, int b) {return (a < b) ?a :b;}

/*****************************************************************************
 *
 *  tdpAtomicAddInt
 *
 *****************************************************************************/

__device__ int tdpAtomicAddInt(int * sum, int val) {

  int old;

  assert(sum);

#ifdef _OPENMP
  /* Some compilers dislike capture: use #pragma omp critical(atomicAddInt) */
  #pragma omp atomic capture
  {
    old = *sum;
    *sum += val;
  }
#else
  old = *sum;
  *sum += val;
#endif

  return old;
}

/*****************************************************************************
 *
 *  tdpAtomicMaxInt
 *
 *  maxval expected to be __shared__
 *
 *****************************************************************************/

__device__ int tdpAtomicMaxInt(int * maxval, int val) {

  int old;

  assert(maxval);

#ifdef _OPENMP
  /* Ug. */
  #pragma omp critical (atomicMaxInt)
  {
    old = *maxval;
    *maxval = int_max(*maxval, val);
  }
#else
  old = *maxval;
  *maxval = int_max(*maxval, val);
#endif

  return old;
}

/*****************************************************************************
 *
 *  tdpAtomicMinInt
 *
 *****************************************************************************/

__device__ int tdpAtomicMinInt(int * minval, int val) {

  int old;

  assert(minval);

#ifdef _OPENMP
  #pragma omp critical (atomicMinInt)
  {
    old = *minval;
    *minval = int_min(*minval, val);
  }
#else
  old = *minval;
  *minval = int_min(*minval, val);
#endif

  return old;
}

/*****************************************************************************
 *
 *  tdpAtomicAddDouble
 *
 *****************************************************************************/

__device__ double tdpAtomicAddDouble(double * sum, double val) {

  double old;

  assert(sum);

#ifdef _OPENMP
  /* Some compilers dislike capture: use #pragma omp critical(atomicAddD) */
  #pragma omp atomic capture
  {
    old = *sum;
    *sum += val;
  }
#else
  old = *sum;
  *sum += val;
#endif

  return old;
}

static double double_max(double a, double b) {return (a > b) ?a :b;}
static double double_min(double a, double b) {return (a < b) ?a :b;}

/*****************************************************************************
 *
 *  tdpAtomicMaxDouble
 *
 *****************************************************************************/

__device__ double tdpAtomicMaxDouble(double * maxval, double val) {

  double old;

  assert(maxval);

#ifdef _OPENMP
#pragma omp critical (atomicMaxDouble)
  {
    old = *maxval;
    *maxval = double_max(*maxval, val);
  }
#else
  old = *maxval;
  *maxval = double_max(*maxval, val);
#endif

  return old;
}

/*****************************************************************************
 *
 *  tdpAtomicMinDouble
 *
 *****************************************************************************/

__device__ double tdpAtomicMinDouble(double * minval, double val) {

  double old;

  assert(minval);

#ifdef _OPENMP
  #pragma omp critical (atomicMinDouble)
  {
    old = *minval;
    *minval = double_min(*minval, val);
  }
#else
  old = *minval;
  *minval = double_min(*minval, val);
#endif

  return old;
}

/*****************************************************************************
 *
 *  tdpAtomicBlockAddInt
 *
 *  See, e.g.,
 *  https://devblogs.nvidia.com/parallelforall/
 *                              faster-parallel-reductions-kepler/
 *
 *  The partial sums partsum must be __shared__; they are destroyed
 *  on exit.
 *  The result is only significant at thread zero.
 *
 *****************************************************************************/

__device__ int tdpAtomicBlockAddInt(int * partsum) {

#ifdef _OPENMP
  int istr;
  int nblock;
  int nthread = omp_get_num_threads();
  int idx = omp_get_thread_num();

  nblock = pow(2, ceil(log(1.0*nthread)/log(2)));

  for (istr = nblock/2; istr > 0; istr /= 2) {
    #pragma omp barrier
    if (idx < istr && idx + istr < nthread) {
      partsum[idx] += partsum[idx + istr];
    }
  }
#endif

  return partsum[0];
}

/*****************************************************************************
 *
 *  tdpAtomicBlockAddDouble
 *
 *****************************************************************************/

__device__ double tdpAtomicBlockAddDouble(double * partsum) {

#ifdef _OPENMP
  int istr;
  int nblock;
  int nthread = omp_get_num_threads();
  int idx = omp_get_thread_num();

  nblock = pow(2, ceil(log(1.0*nthread)/log(2)));

  for (istr = nblock/2; istr > 0; istr /= 2) {
    #pragma omp barrier
    if (idx < istr && idx + istr < nthread) {
      partsum[idx] += partsum[idx + istr];
    }
  }
#endif

  return partsum[0];
}

