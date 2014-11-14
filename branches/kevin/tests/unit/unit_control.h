/*****************************************************************************
 *
 *  control.h
 *
 *  Test control.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef CONTROL_H
#define CONTROL_H

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>

#include "e4c_mpi.h"

typedef struct control_s control_t;

typedef enum control_enum {CONTROL_PASS, 
			   CONTROL_FAIL,
			   CONTROL_QUIET,
                           CONTROL_VERBOSE} control_enum_t;

int control_create(MPI_Comm comm, control_t ** ctrl);
int control_free(control_t * ctrl);
int control_comm(control_t * ctrl, MPI_Comm * comm);
int control_test(control_t * ctrl, const char * file, const char * func);
int control_verb(control_t * ctrl, const char * fmt, ...);
int control_option_set(control_t * ctrl, control_enum_t key);
int control_report(control_t * ctrl);
int control_fail_line_number_set(control_t * ctrl, int line_number);
int control_assert_dbl(double d1, double d2, double tol);
int control_allfail(control_t * ctrl);

#define __CONTROL_INFO__ __FILE__, __FUNCTION__


#define control_macro_assert(ctrl, expr, except)      \
  do { if ( !(expr) ) {				      \
    control_fail_line_number_set(ctrl, __LINE__); \
    throw (except, ""); \
    } \
  } while(0)

#define control_macro_eq(d1, d2, tol) ( fabs((d1) - (d2)) < (tol) )

#define control_macro_test(ctrl, expr) \
  do { if ( !(expr) ) {				      \
    control_fail_line_number_set(ctrl, __LINE__); \
    throw (TestFailedException, ""); \
    } \
  } while(0)

#define control_macro_test_dbl_eq(ctrl, d1, d2, tol) \
  do { if (!control_macro_eq((d1), (d2), (tol)) ) {	   \
      control_fail_line_number_set(ctrl, __LINE__);	   \
      throw(TestFailedException, "");				   \
    } \
  } while (0)

#endif
