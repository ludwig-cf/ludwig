
#include <assert.h>
#include <stdio.h>

#include "object1_kernel.h"

__target_entry__ void object1_kernel(object1_t * obj1);


__host__ int object1_kernel_driver(object1_t * obj1) {

  int data;
  object1_t * obj1_target;

  assert(obj1);
  printf("Hello from obj1 kernel driver\n");
  data = object1_function(obj1);
  printf("Host obj1 data: %d\n", data);

  object1_target(obj1, &obj1_target);
  target_launch(object1_kernel, 1, 1, obj1_target);

  return 0;
}

__target_entry__ void object1_kernel(object1_t * obj1) {

  int data;

  printf("Hello from obj1 kernel\n");

  data = object1_function(obj1);

  printf("Data obj1 kernel %d\n", data);

  return;
}
