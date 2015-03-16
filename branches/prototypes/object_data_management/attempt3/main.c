/* OpenMP trial */

#include <assert.h>
#include <stdio.h>

#include "targetDP.h"
#define NDATA 4
#define SIMDVL 2


#ifdef _OPENMP
  /* have OpenMP */
  #include <omp.h>
#else
  /* NULL OpenMP implmentation */
  #define omp_get_thread_num()  0
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
#endif


#ifdef __NVCC__
/* target defs, including real target_simd_loop() */
#define target_simt_region()
#define target_simt_loop(index, ndata, simdvl) \
  index = 0;
#define __simt_builtin_init()

#else

/* Dummy */

#define __shared__

/* Dummy built-in variable implmentation. */

typedef struct uint3_s uint3;
typedef uint3 uint3_t;
struct uint3_s {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

uint3_t __simd_builtin_threadIdx_init(void) {
  uint3_t init = {1, 1, 1};
  init.x = omp_get_thread_num();
  return init;
}

uint3_t __simd_builtin_blockDim_init(void) {
  uint3_t init = {1, 1, 1};
  init.x = omp_get_num_threads();
  return init;
}
/* Within target_simd_region(), provide access/initialisation */
#define __simt_builtin_init() \
  uint3_t threadIdx = __simd_builtin_threadIdx_init();	\
  uint3_t blockDim  = __simd_builtin_blockDim_init();

/* data parallel OpenMP */

#define target_simt_region() _Pragma("omp parallel")

#define target_simt_loop(index, ndata, simdvl) \
  _Pragma("omp for nowait") \
  for (index = 0; index < ndata; index += simdvl)


#endif

__target_entry__ void kernel_trial(int ndata, double * data);

int main(int argc, char ** argv) {

  int ndata = NDATA;
  int n;
  double hdata[NDATA];
  double * data;

  for (n = 0; n < ndata; n++) {
    hdata[n] = 1.0*n;
  }

  targetMalloc((void **) &data, 5*sizeof(double));
  /*targetMemcpy();*/
  data = hdata;


  target_launch(kernel_trial, 1, 1, ndata, data);
  syncTarget();

  /* targetMemcpy */

  printf("ndara is %d\n", ndata);
  for (n = 0; n < ndata; n++) {
    printf("data[%2i] = %5.1f\n", n, data[n]);
  }

  return 0;
}

__target_entry__ void kernel_trial(int ndata, double * data) {

  int index;                     /* Problem: shared in host implementation */
  __shared__ double sdata[10];   /* OK; shared in host/device */

  assert(ndata % SIMDVL == 0);

  target_simt_region() {
    /* Declare thread-private variables if required; here... */

    target_simt_loop(index, ndata, SIMDVL) {

      /* ... or here  */

      /* build-in variables can be accessed only if initialised via ... */
      __simt_builtin_init();

      printf("Thread %d of %d index %d\n", threadIdx.x, blockDim.x, index);

      if (index < ndata) {

	int iv;    /* index for simd loop private */

	for (iv = 0; iv < SIMDVL; iv++) {
	  /*printf("Update simd %d\n", index + iv);*/
	  data[index + iv] *= 2.0;
	}
      }
    }
  }

  return;
}
