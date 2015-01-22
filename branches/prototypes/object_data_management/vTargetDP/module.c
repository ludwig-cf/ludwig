
#include <stdio.h>

#include "module.h"
#include "targetDP.h"


//these need to be allocated statically
static kernel_const_t static_const;
static TARGET_CONST kernel_const_t t_static_const;

HOST void object_create(obj_t ** pobj, int c1in, int c2in){

  obj_t * obj = NULL;

  obj = (obj_t*) calloc(1, sizeof(obj_t));

  //associate constant part of object with statically allocated space:

  //host
  obj->const_host=&static_const;

  //target
  void * ptrtmp;
  //get constant memory address from target
  __getTargetConstantAddress__(&ptrtmp, t_static_const); 
  obj->const_target=(kernel_const_t*) ptrtmp; 

  // set host copy
  obj->const_host->c1=c1in;
  obj->const_host->c2=c2in;

  // propagate to target copy
  __copyConstantToTarget__(&t_static_const,obj->const_host, sizeof(kernel_const_t));

  // allocate data on target
  targetMalloc((void **) &(obj->data_target.field),N*sizeof(double));

  *pobj = obj;

}

HOST void object_free(obj_t * obj){

  targetFree(obj->data_target.field);

}
