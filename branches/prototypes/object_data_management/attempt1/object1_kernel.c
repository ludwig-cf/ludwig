
#include <assert.h>
#include <stdio.h>

#include "object1_kernel.h"

__target_entry__ void object1_kernel(object1_t * obj1);


__host__ int object1_kernel_driver(object1_t * obj1) {

  int data;
  object1_t * obj1_target;

  assert(obj1);
  data = object1_function(obj1);

  printf("Host result obj1 data: %d\n", data);

  object1_target(obj1, &obj1_target);
  target_launch(object1_kernel, 1, 1, obj1_target);

  syncTarget();
  checkTargetError("obj1 kernel ");

  return 0;
}

__target_entry__ void object1_kernel(object1_t * obj1) {

  int data;

  data = object1_function(obj1);

  printf("Kernel result obj1 data: %d\n", data);

  return;
}
