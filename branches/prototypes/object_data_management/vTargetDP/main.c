
#include <stdio.h>
#include "module.h"
#include "targetDP.h"

void kernel_wrapper(kernel_data_t d_array, kernel_const_t* const_ptr);

int main(){

  
  obj_t myobj;

  //set constants to arbitrary values 999 and 25
  const_init(&myobj,999,25);

  field_init(&myobj);

  kernel_wrapper(myobj.data_target,myobj.const_target);

  field_finalise(&myobj);

  return 0;

}
