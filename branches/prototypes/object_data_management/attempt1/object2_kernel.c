
#include <stdio.h>

#include "object2_s.h"

__target_entry__ void object2_kernel(object2_t * obj2);


__host__ int object2_kernel_driver(object2_t * obj2) {

  int data;

  printf("Hello from obj2 kernel driver\n");
  data = object2_function(obj2);
  printf("Host obj2 result: data = %d\n", data);

  target_launch(object2_kernel, 1, 1, obj2->target);

  return 0;
}

__target_entry__ void object2_kernel(object2_t * obj2) {

  int data;

  printf("Hello from obj2 kernel\n");

  data = object2_function(obj2);

  printf("Kernel obj2 result = %d\n", data);

  return;
}
