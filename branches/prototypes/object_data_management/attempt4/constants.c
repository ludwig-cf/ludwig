/* Address device constant memory via pointer check */

#include <assert.h>
#include <stdio.h>

#include "targetDP.h"

typedef struct data_s data_t;
typedef struct my_data_s my_data_t;

struct data_s {
  int ia;
  int ib;
  double b;
};

struct my_data_s {
  data_t * data;
  my_data_t * target;
};

static __constant__ data_t my_static_data;


int my_data_create(my_data_t ** pobj);
int my_data_free(my_data_t * obj);
int my_data_init1(my_data_t * obj);
int my_data_test(my_data_t * obj);
__global__ void my_init1_kernel(my_data_t * obj);
__global__ void my_data_test_kernel(my_data_t * obj);



int main(int argc, char ** argv) {

  my_data_t * data = NULL;

  my_data_create(&data);
  my_data_test(data);

  /* Update */
  data->data->ia = 4;
  data->data->b  = 5.0;
  copyConstToTarget(&my_static_data, data->data, sizeof(data_t));
  my_data_test(data);


  my_data_free(data);

  return 0;
}


int my_data_create(my_data_t ** pobj) {

  my_data_t * obj = NULL;

  obj = (my_data_t *) calloc(1, sizeof(my_data_t));
  assert(obj);

  obj->data = (data_t *) calloc(1, sizeof(data_t));
  assert(obj->data);

  targetCalloc((void **) &obj->target, sizeof(data_t));
  my_data_init1(obj);

  *pobj = obj;

  return 0;
}

int my_data_free(my_data_t * obj) {


  targetFree(obj->target);
  free(obj);

  return 0;
}

/* Initialise data */

int my_data_init1(my_data_t * obj) {

  dim3 nblk, ntpb;
  data_t * here = NULL;

  nblk.x = 1; nblk.y = 1; nblk.z = 1;
  ntpb.x = 1; ntpb.y = 1; ntpb.z = 1;

  obj->data->ia = 3;
  obj->data->b  = 4.0;

  copyConstToTarget(&my_static_data, obj->data, sizeof(data_t));

  /* Run a kernel */

  /*
  __host_launch_kernel(my_init1_kernel, nblk, ntpb, obj->target);
  targetDeviceSynchronise();
  */

  /* Or, copy the symbol address */

  cudaGetSymbolAddress((void **) &here, my_static_data);
  cudaMemcpy(&obj->target->data, (const void *) &here, sizeof(data_t *),
	     cudaMemcpyHostToDevice);

  return 0;
}

/* Initialise device memory pointer kernel */

__global__
void my_init1_kernel(my_data_t * obj) {

  obj->data = &my_static_data;

  return;
}

/* Test driver */

int my_data_test(my_data_t * obj) {

  dim3 nblk, ntpb;

  nblk.x = 1; nblk.y = 1; nblk.z = 1;
  ntpb.x = 1; ntpb.y = 1; ntpb.z = 1;

  __host_launch_kernel(my_data_test_kernel, nblk, ntpb, obj->target);
  targetDeviceSynchronise();

  return 0;
}

/* Test kernel */

__global__
void my_data_test_kernel(my_data_t * obj) {

  printf("Constant ia %d\n", my_static_data.ia);
  printf("Constant b  %f\n", my_static_data.b);
  printf("Kernel data->ia %d\n", obj->data->ia);
  printf("Kernel data->b  %f\n", obj->data->b);

  return;
}

