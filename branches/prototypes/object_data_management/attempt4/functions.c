#include <assert.h>
#include <stdio.h>
#include "targetDP.h"

/* Host device function pointers */
typedef struct fe_s fe_t;

typedef int (* fe_host_ft)(fe_t * fe, fe_t ** target);
typedef int (* fe_host_dev_ft)(fe_t * fe, const char * msg);
typedef int (* fe_dev_ft)(fe_t * fe, const char * msg);

typedef struct fe_vt_s fe_vt_t;

struct fe_vt_s {
  fe_host_ft f1;
  fe_host_dev_ft f2;
  fe_dev_ft f3;
};

struct fe_s {
  fe_vt_t * child;
};

/* Concrete implementation */

typedef struct fe_symm_s fe_symm_t;
__host__ int fe_symm_target(fe_symm_t * fe, fe_t ** target);
__host__ __device__ int fe_symm_host_target(fe_symm_t * fe, const char * msg);
__device__ int fe_symm_dev(fe_symm_t * fe, const char * msg);
__global__ void kernel_function(fe_t * fe);

struct fe_symm_s {
  fe_t parent;
  /* Other stuff */
  fe_symm_t * target;
};

static fe_vt_t fe_symm_hvt = {
  (fe_host_ft) fe_symm_target,
  (fe_host_dev_ft) fe_symm_host_target,
  (fe_dev_ft) NULL
};

static __constant__ fe_vt_t fe_symm_dvt = {
  (fe_host_ft) NULL,
  (fe_host_dev_ft) fe_symm_host_target,
  (fe_dev_ft) fe_symm_dev
};

/* Start implementation */

__host__ int fe_symm_create(fe_symm_t ** p) {

  int ndevice;
  fe_symm_t * fes = NULL;

  fes = (fe_symm_t *) calloc(1, sizeof(fe_symm_t));
  assert(fes);

  fes->parent.child = &fe_symm_hvt; /* Host vtable */

  /* Device */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fes->parent.child->f3 = (fe_dev_ft) fe_symm_dev; /* host is target */
    fes->target = fes;
  }
  else {
    fe_vt_t * tmp;
    /* Allocate and set device vtable */
    targetCalloc((void **) &fes->target, sizeof(fe_symm_t));
    targetConstAddress(&tmp, fe_symm_dvt);
    copyToTarget(&fes->target->parent.child, tmp, sizeof(fe_vt_t *));
  }

  *p = fes;

  return 0;
}

/* Virtual function implementations */

__host__ int fe_symm_target(fe_symm_t * fe, fe_t ** target) {

  *target = (fe_t *) fe->target;

  return 0;
}

__host__ __device__ int fe_symm_host_target(fe_symm_t * fe, const char * msg) {

  printf("Hello from __host__ __device__ %s\n", msg);

  return 0;
}

__device__ int fe_symm_dev(fe_symm_t * fe, const char * msg) {

  printf("Hello from dev function %s\n", msg);

  return  0;
}

int main(int argc, char ** argv) {

  dim3 nblk, ntpb;
  fe_t * fe = NULL;
  fe_symm_t * fes = NULL;

  fe_symm_create(&fes);
  assert(fes);

  fe = (fe_t *) fes;
  fe->child->f2(fe, "called from host");

  fe = NULL;
  nblk.x = 1; nblk.y = 1; nblk.z = 1;
  ntpb.x = 1; ntpb.y = 1; ntpb.z = 1;
  fe_symm_target(fes, &fe);

  __host_launch(kernel_function, nblk, ntpb, fe);

  return 0;
}

__global__ void kernel_function(fe_t * fe) {

  fe->child->f2(fe, "called from kernel");
  printf("Calling f3\n");
  fe->child->f3(fe, "called from kernel");

  return;
}
