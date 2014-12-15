
#include <stdio.h>

#include "module.h"

//these need to be allocated statically
static obj_const_data static_const;
static __constant__ obj_const_data t_static_const;

void checkCUDAError(const char *msg);

void const_init(obj* myobj, int c1in, int c2in){

 
  //associate constant part of object with statically allocated space
  myobj->const_host=&static_const; //host

  // get constant memory address from GPU
  void * ptrtmp;
  cudaGetSymbolAddress(&ptrtmp, t_static_const); //target
  checkCUDAError("getsymboladdress");
  myobj->const_target=(obj_const_data*) ptrtmp; 

  //now, myobj.const_host and myobj.const_target point to the right locations


  // set host version
  myobj->const_host->c1=c1in;
  myobj->const_host->c2=c2in;


  // copy to target version
  cudaMemcpyToSymbol(t_static_const, myobj->const_host, sizeof(obj_const_data), 0,cudaMemcpyHostToDevice);
  checkCUDAError("memcpytosymbol");

}

void field_init(obj* myobj){

  cudaMalloc(&(myobj->data_target.field),N*sizeof(double));


}


void field_finalise(obj* myobj){

  cudaFree(myobj->data_target.field);

}
