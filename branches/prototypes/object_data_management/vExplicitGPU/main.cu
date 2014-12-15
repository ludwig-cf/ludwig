
#include <stdio.h>
#include "module.h"

void kernel_wrapper(obj_kernel_data d_array, obj_const_data* const_ptr);

int main(){

  
  obj myobj;

  //set constants to arbitrary values 999 and 25
  const_init(&myobj,999,25);

  field_init(&myobj);

  kernel_wrapper(myobj.data_target,myobj.const_target);

  field_finalise(&myobj);

  return 0;

}
