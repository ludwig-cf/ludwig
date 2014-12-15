
#include <stdio.h>
#include "module.h"
#include "targetDP.h"

void kernel_wrapper(kernel_data_t d_array, kernel_const_t* const_ptr);

int main(){

  
  obj_t* myobj = NULL;

  //create object and set constants to arbitrary values 999 and 25
  object_create(&myobj,999,25);

  kernel_wrapper(myobj->data_target,myobj->const_target);

  object_free(myobj);
    
  return 0;

}
