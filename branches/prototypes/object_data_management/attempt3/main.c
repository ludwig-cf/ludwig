/* OpenMP trial */

#include <assert.h>
#include <stdio.h>

#include "targetDP.h"
#define NDATA 4
#define SIMDVL 2


#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num()  0
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
#endif


#define target_simt_region() _Pragma("omp parallel")
#define target_simt_loop(index, ndata, simdvl) \
  _Pragma("omp for nowait") \
  for (index = 0; index < ndata; index += simdvl)

/* To directly follow target_simt_region() to make threadIdx etc available */
#define __simt_builtin_init() \
  int threadIdx = omp_get_thread_num();		 \
  int blockDim  = omp_get_num_threads();


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

  int index;    /* index for thread loop declared ahead of time  */
  /* __shared__ declaration possible before simt region */

  assert(ndata % SIMDVL == 0);

  target_simt_region() {
    /* Declare thread-private variables if required; here... */

    target_simt_loop(index, ndata, SIMDVL) {

      /* ... or here  */

      /* build-in variables can be accessed only if initialised via ... */
      __simt_builtin_init();

      printf("Thread %d of %d index %d\n", threadIdx, blockDim, index);

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
