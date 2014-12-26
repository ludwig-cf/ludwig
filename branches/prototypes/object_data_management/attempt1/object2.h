/* See object2_s.h for comments */

#ifndef OBJECT2_H
#define OBJECT2_H

#include "object1.h"

typedef struct object2_data_s object2_data_t;
typedef struct object2_s object2_t;

__host__ int object2_create(object1_t * obj1, object2_t ** p);
__host__ int object2_free(object2_t * obj2);
__host__ int object2_memcpy(object2_t * obj2, int kind);

__host__ __device__ int object2_function(object2_t * obj2);

#endif
