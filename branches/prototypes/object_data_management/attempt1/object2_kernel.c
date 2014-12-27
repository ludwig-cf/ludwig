
#include <assert.h>
#include <stdio.h>

#include "object2_s.h"

__target_entry__ void object2_kernel(object2_t * obj2);

__host__ int object2_kernel_driver(object2_t * obj2) {

  int data;
  object2_t * target = NULL;

  data = object2_function(obj2);

  printf("Host result obj2 data: %d\n", data);

  object2_target(obj2, &target);

  target_launch(object2_kernel, 1, 1, target);

  syncTarget();
  checkTargetError("obj2 kernel ");

  return 0;
}

__target_entry__ void object2_kernel(object2_t * obj2) {

  int data = 0;

  data = object2_function(obj2);

  printf("Kernel result obj2 data: %d\n", data);

  return;
}
