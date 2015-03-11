
/* Model for struct const on host (this is lbmodel_t here)
 * and therefore suitable for const memory on device */

#include <stdio.h>

#include "targetDP.h"

typedef struct lbmodel_s lbmodel_t;
typedef struct kernel_data_s kernel_data_t;

#define NVEL 1

struct kernel_data_s {
  double * data1;
  double * data2;
};

struct lbmodel_s {
  int cv[NVEL][3];
  int iw[NVEL];
};

lbmodel_t const lbmodel = {
  /* cv: basis vectors (integers) */
  {{1, 2, 3}},
  /* iv: weight x 36 (integers) */
  {36}
};


/* nvcc (version) expands __constant__ as something
 * which causes X error if &lbmodel appears on the right hand side;
 * a cast to void can be used (CHECK). */

const lbmodel_t * const host = &lbmodel;
__constant__ lbmodel_t * const target = (void *) &lbmodel;



__target_entry__ void kernel_function(kernel_data_t * obj);
__host__ __device__ int test_function(kernel_data_t * obj);


int main(int argc, char ** argv) {

  kernel_data_t obj;

  printf("MAIN: ma.a[1] = %d\n", host->cv[0][0]);
  printf("MAIN: sizeof(lb) =%lu\n", sizeof(lbmodel_t));

  target_launch(kernel_function, 1, 1, &obj);

  return 0;
}

__target_entry__ void kernel_function(kernel_data_t * obj) {

  printf("kernel function: %d\n", target->cv[0][0]);
  test_function(obj);

  return;
}

__host__ __device__ int test_function(kernel_data_t * obj) {

  double * restrict data1 = obj->data1;
  double * restrict data2 = obj->data2;
  lbmodel_t const * const restrict lbdata = target;

  data1 = NULL;
  data2 = NULL;

  printf("Host device function: %d\n", lbdata->cv[0][0]);

  return 0;
}
