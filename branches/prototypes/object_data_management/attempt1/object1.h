/* See object1_s.h for comments */

#ifndef OBJECT1_H
#define OBJECT1_H

#include "targetDP.h"

typedef struct object1_data_s object1_data_t;
typedef struct object1_s object1_t;

__host__ int object1_create(int data, object1_t ** phost);
__host__ int object1_free(object1_t * obj1);
__host__ int object1_target(object1_t * obj1, object1_t ** ptarget);

__host__ __device__ int object1_function(object1_t * obj1);

#endif
