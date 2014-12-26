
#include <assert.h>
#include <stdio.h>

#include "object1_s.h"

static int object1_device_create(object1_t * self, object1_t ** pdevice);
static int object1_device_free(object1_t * obj);


__host__ int object1_create(int data1, object1_t ** p) {

  object1_t * obj = NULL;

  assert(p);

  obj = (object1_t *) calloc(1, sizeof(object1_t));
  assert(obj);

  obj->data = (object1_data_t *) calloc(1, sizeof(object1_data_t));
  assert(obj->data);

  object1_device_create(obj, &obj->device);

  /* Initial data */
  obj->data->data1 = data1;

  *p = obj;

  return 0;
}

__host__ int object1_free(object1_t * obj) {

  assert(obj);

  object1_device_free(obj->device);
  free(obj->data);
  free(obj);

  return 0;
}

__host__ int object1_device_create(object1_t * self, object1_t ** pdevice) {

  assert(self);
  assert(pdevice);

  if (target_is_host()) {
    *pdevice = self;
  }
  else {
    targetMalloc((void **) pdevice, sizeof(object1_t));
    targetMalloc((void **) &(*pdevice)->data, sizeof(object1_data_t));
    /* Allocate data->data2 if required */
  }

  return 0;
}

__host__ int object1_device_free(object1_t * device) {

  assert(device);

  if (target_is_host()) {
    /* No action */
  }
  else {
    targetFree(device->data);
    targetFree(device);
  }

  return 0;
}

__host__ int object1_target(object1_t * obj1, object1_t ** ptarget) {

  assert(obj1);
  assert(obj1->device);

  *ptarget = obj1->device;

  return 0;
}

__host__ __device__ int object1_function(object1_t * obj) {

  int idata;

  idata = obj->data->data1;

  return idata;
}
