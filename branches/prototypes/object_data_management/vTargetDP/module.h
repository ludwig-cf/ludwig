
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


void const_init(obj_t * obj, int c1in, int c2in);
void field_init(obj_t * obj);
void field_finalise(obj_t * obj);
