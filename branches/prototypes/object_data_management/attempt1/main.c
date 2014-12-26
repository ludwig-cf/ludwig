
#include <stdio.h>

#include "object1_kernel.h"
#include "object2_kernel.h"

int main(int argc, char * argv[]) {

  object1_t * obj1 = NULL;
  object2_t * obj2 = NULL;

  printf("Hello form main()\n");

  object1_create(3, &obj1);
  object2_create(obj1, &obj2);

  object1_kernel_driver(obj1);
  object2_kernel_driver(obj2);

  object2_free(obj2);
  object1_free(obj1);

  printf("Goodbye from main()\n");

  return 0;
}
