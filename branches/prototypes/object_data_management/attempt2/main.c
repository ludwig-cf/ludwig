
/*
 * A model problen for obtaining __constant__ memory in
 * __host__ __device__ functions.
 *
 * For this to work, both host and device must reference
 * the appropriate variable, i.e., host memory on host version
 * and device memory in device version. To maintain a single
 * source "__host__ __deivice__" function, this means that a
 * reference should be pased via the argument list.
 *
 * A solution is to keep a pointer to __constant__ memory
 * in kernel arguments which can be set appropriately on
 * host and device.
 */

#include <stdio.h>
#include <stdlib.h>

#include "targetDP.h"

typedef struct lbmodel_s lbmodel_t;
typedef struct kernel_data_s kernel_data_t;

/*
 * Strucuture for constant data.
 *
 * Any initialiser must be literal constant. E.g.,
 *
 * Error: __constant__ lbmodel_t * target = &lbmodel_variable;
 *
 */

struct lbmodel_s {
  int cv[1][3];
  int iw[1];
};

#define lbmodel_initialiser() {{{-1, 0, 0}}, {18}}

__constant__ __device__ int idata = 3;
__constant__ __device__ int ivect[3] = {0, 1, 2};

const lbmodel_t host_data = lbmodel_initialiser();
__constant__ __device__ lbmodel_t target_data = lbmodel_initialiser();

/* Kernel data structure (host and device) */

struct kernel_data_s {
  double * data1;              /* not used */
  lbmodel_t const * lbconst;   /* pointer to const data */
  kernel_data_t * target;      /* device copy */
};

__target_entry__ void target_constant(kernel_data_t * obj);
__target_entry__ void kernel_function(kernel_data_t * obj);
__host__ __device__ int test_function(kernel_data_t * obj);


int main(int argc, char ** argv) {

  kernel_data_t * obj;

  /* host structure */
  obj = (kernel_data_t *) calloc(1, sizeof(kernel_data_t));
  obj->lbconst = &host_data;

  /* device memory */
  if (target_is_host()) {
    obj->target = obj;
  }
  else {
    targetMalloc((void **) &obj->target, sizeof(kernel_data_t));
    checkTargetError("Malloc problem");

    /* The only (repeat only) way to get the address of the __constant__
     * data for use on the device is via a kernel on the device.
     * Any cudaGetSymbolAddress() etc provides addresses only valid
     * on the host */

    target_launch(target_constant, 1, 1, obj->target);
    syncTarget();
  }

  printf("host: data->cv[0][0] = %d data->iw[0] = %i\n", host_data.cv[0][0],
	 host_data.iw[0]);
  printf("host: sizeof(lb) =%lu\n", sizeof(lbmodel_t));

  target_launch(kernel_function, 1, 1, obj->target);
  syncTarget();
  checkTargetError("Kernel launch");

  printf("No clean-up, but normal exit\n");

  return 0;
}

__target_entry__ void target_constant(kernel_data_t * obj) {

  obj->lbconst = &target_data;

  return;
}

__target_entry__ void kernel_function(kernel_data_t * obj) {

  printf("In kernel function: %d\n", obj->lbconst->cv[0][0]);
  test_function(obj);

  return;
}

__host__ __device__ int test_function(kernel_data_t * obj) {

  double * __restrict__ data1 = obj->data1;
  lbmodel_t const * __restrict__ lb = obj->lbconst;

  data1 = NULL;

  printf("data1 is %u\n", data1);
  printf("In host device function: %d %i\n", lb->cv[0][0], lb->iw[0]);

  /*
   * Error: not available for __host__ version
   * printf("Host device function: %d\n", idata);
   * printf("Host device function: %d %d %d\n", ivect[0], ivect[1], ivect[2]);
   */

  return 0;
}
