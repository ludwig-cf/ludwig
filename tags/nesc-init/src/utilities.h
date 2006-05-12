#ifndef _UTILITIES_H
#define _UTILITIES_H

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>
#include<float.h>

/*--------------------------Standard typedefs--------------------------------*/

typedef double Float;

typedef struct{
  Float x,y,z;
} FVector;

typedef struct{
  int x,y,z;
} IVector;

typedef struct{
  FVector x,y,z;
} FTensor;



/* Doubly linked list to manipulate input strings */
typedef struct input_data Input_Data;
struct input_data{
  char str[256];
  Input_Data *last,
    *next;
};

enum io_type { BINARY, ASCII };

/* struct for parallel IO */
typedef struct{
  int root;                      /* Root PE of current I/O group (TRUE or
				    FALSE) */
  int n_io;                      /* Number of parallel IO group */
  int size;                      /* Size (in PEs) of each IO group */
  int index;                     /* Index of current IO group (start at 1) */
  int rank;                      /* Rank of PE in IO group */
  char *file_ext;                /* Filename suffix for parallel output */
} IO_Param;


/*--------------------- Few standard defines and macros, etc ----------------*/

#define PI 3.1415926535897932385E0
#define TINY 1.0E-15

/* Enums always start at 0, then 1, etc. */
/* Do *NOT* change the order: FLUID has to be the 1st term, i.e., set to 0 */
/* (KS: why? ... probably because never explicitly initialised.) */

#ifdef _HPCX_
/* Redefining FALSE, TRUE causes indigestion so ... */
#undef FALSE
#undef TRUE
#endif

enum { FALSE, TRUE };

enum { FLUID, SOLID, COLLOID };

/* Tracing macros */
#define LUDWIG_MSG_PREFIX "LUDWIG("__FILE__")"

#define LMP LUDWIG_MSG_PREFIX
#ifdef _TRACE_
#define LUDWIG_ENTER(func_name) \
        fprintf(stderr, LMP"[TRACE] Entering %s\n", func_name);
#else
#define LUDWIG_ENTER(func_name)
#endif

/*--------------------- Utility routines (timing, etc) ----------------------*/

Float   UTIL_fdistance_sq(FVector r1, FVector r2);
Float   UTIL_fvector_mod(FVector r);
Float   UTIL_dot_product(FVector v1, FVector v2);
FVector UTIL_fvector_zero(void);
FVector UTIL_fvector_add(FVector v1, FVector v2);
FVector UTIL_fvector_subtract(FVector v1, FVector v2);
FVector UTIL_cross_product(FVector v1, FVector v2);
FVector UTIL_rotate_vector(FVector, FVector);

/*--------------------- End of header file ----------------------------------*/
#endif /* _UTILITIES_H */
