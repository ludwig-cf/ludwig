
#include "targetDP.h"

#define N 4
 
typedef struct {
 double * field;
} kernel_data_t;

// N.B. kernel_const_t is defined in targetDP.h

typedef struct {
  kernel_const_t * const_host; 
  kernel_const_t * const_target; 
  kernel_data_t data_host;
  kernel_data_t data_target;
  /* Additional data only relevant for host:  MPI, I/O, etc ... */
} obj_t;


HOST void object_create(obj_t ** pobj, int c1in, int c2in);
HOST void object_free(obj_t * obj);
