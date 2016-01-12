/*
 * simleExample.c: simple targetDP example
 * Alan Gray
 * 
 * Copyright 2015 The University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * to compile for CPU:
 * gcc -I. -c simpleExample.c
 * gcc -I. simpleExample.c libtargetDP_C.a 
 * 
 * to compile for GPU:
 * nvcc -x cu -arch=sm_35 -I. -DCUDA -dc -c simpleExample.c 
 * nvcc -arch=sm_35 simpleExample.o libtargetDP_CUDA.a 
 *
 */

#include <stdio.h>
#include <targetDP.h>

#define N 1024

__targetConst__ double t_a;

__targetEntry__ void scale(double* t_field) {

   int baseIndex;
   __targetTLP__(baseIndex, N) {

    int iDim, vecIndex;
    for (iDim = 0; iDim < 3; iDim++) {

         __targetILP__(vecIndex)  \
         t_field[iDim*N + baseIndex + vecIndex] =  \
	  t_a*t_field[iDim*N + baseIndex + vecIndex];      	  
    }
  }
  return;
}


int main(){


  double* field;
  double* t_field;
  
  double a=1.23;
  
  int i;

  size_t datasize=N*3*sizeof(double);

  field = (double*) malloc(datasize);
  for (i=0;i<N;i++) field[i]=i;
			   
  targetMalloc((void **) &t_field, datasize);
  
  copyToTarget(t_field, field, datasize);
  copyConstToTarget(&t_a, &a, sizeof(double)); 
  
  scale __targetLaunch__(N) (t_field);
  targetSynchronize();
  
  copyFromTarget(field, t_field, datasize);
  targetFree(t_field);


  printf("field[4]=%f\n",field[4]);
  free(field);
			   

  return 0;

}

