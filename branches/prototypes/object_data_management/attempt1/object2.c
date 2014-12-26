
#include <assert.h>
#include <stdio.h>

#include "object2_s.h"

static __host__
int object2_target_create(object2_t * self, object2_t ** ptarget);
static __host__
int object2_target_free(object2_t * obj2);


__host__
int object2_create(object1_t * obj1, object2_t ** p) {

  object2_t * obj2 = NULL;

  assert(obj1);
  assert(p);

  obj2 = (object2_t *) calloc(1, sizeof(object2_t));
  assert(obj2);

  obj2->data = (object2_data_t *) calloc(1, sizeof(object2_data_t));
  assert(obj2->data);

  object2_target_create(obj2, &obj2->target);

  /* For data we just use a hardwired example; host reference to obj1 */
  obj2->data->datum = 5;
  obj2->obj1 = obj1;

  object2_memcpy(obj2, 0);

  *p = obj2;

  return 0;
}

__host__
int object2_free(object2_t * obj2) {

  assert(obj2);

  object2_target_free(obj2->target);
  free(obj2->data);
  free(obj2);

  return 0;
}

static __host__
int object2_target_create(object2_t * self, object2_t ** p) {

  if (target_is_host()) {
    *p = self;
  }
  else {
    targetMalloc((void **) p, sizeof(object2_t));
    targetMalloc((void **) &(*p)->data, sizeof(object2_data_t));
  }

  return 0;
}

static __host__
int object2_target_free(object2_t * obj2) {

  if (target_is_host()) {
    /* No action */
  }
  else {
    targetFree(obj2->data);
    targetFree(obj2);
  }

  return 0;
}

__host__ int object2_memcpy(object2_t * obj2, int kind) {

  object1_t * obj1;

  if (target_is_host()) {
    /* No action */
  }
  else {
    switch (kind) {
    case 0:
      copyToTarget(obj2->target->data, obj2->data, sizeof(object2_data_t));
      /* We need the device reference from obj1 */
      object1_target(obj2->obj1, &obj1);
      copyToTarget(obj2->obj1, obj1, sizeof(object1_t *));
      break;
    case 1:
      copyFromTarget(obj2->data, obj2->target->data, sizeof(object2_data_t));
      break;
    }
  }

  return 0;
}

__host__ __device__
int object2_function(object2_t * obj2) {

  int data1;
  int data2;

  /* should give e.g., "3x5=15" */

  data1 = object1_function(obj2->obj1);
  data2 = data1*obj2->data->datum;

  return data2;
}
