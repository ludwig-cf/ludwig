#include <stdio.h>
#include "module.h"
#include "targetDP.h"

TARGET_ENTRY void kernel (kernel_data_t kernel_data, kernel_const_t* const_ptr){


  int index;
  TARGET_TLP(index,1){
    printf("Setting first 2 components of d_array to constants:\n");
    kernel_data.field[0]=const_ptr->c1;
    kernel_data.field[1]=const_ptr->c2;
    
    printf("d_array[0]= %f\n",kernel_data.field[0]);
    printf("d_array[1]= %f\n",kernel_data.field[1]);
  }
}


void kernel_wrapper(kernel_data_t kernel_data, kernel_const_t* const_ptr){

  printf("before kernel\n");

  kernel TARGET_LAUNCH(1) (kernel_data, const_ptr);
  
  syncTarget();
  
  checkTargetError("kernel");
  printf("after kernel\n");
}

