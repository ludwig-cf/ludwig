
#define N 4


//__constant__ int c=1987;

struct obj_const_data {
int  c1;
int  c2;
};

struct obj_kernel_data {
 double * field;
};


struct obj {
  struct obj_const_data * const_host; 
  struct obj_const_data * const_target; 
  struct obj_kernel_data data_host;
  struct obj_kernel_data data_target;
  /* Additional data only relevant for host:  MPI, I/O, etc ... */
};

void const_init(obj* myobj, int c1in, int c2in);
void field_init(obj* myobj);
void field_finalise(obj* myobj);
