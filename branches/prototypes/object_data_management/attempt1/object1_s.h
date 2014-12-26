/* Object maintaining host references to
 *   (1) its own data (object1_data_t)
 *   (2) some data only relevant on host (e.g., MPI_Comm)
 *   (3) its own device copy (object1_t * device) */

#ifndef OBJECT1_S_H
#define OBJECT1_S_H

#include <mpi.h>
#include "object1.h"

struct object1_data_s {
  int data1;
  double * data2;
};

struct object1_s {
  object1_data_t * data;
  MPI_Comm auxiliary;        /* Auxiliary host data, for example */
  object1_t * device;        /* Host pointer -> device object */
};

#endif
