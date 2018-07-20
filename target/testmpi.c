/*****************************************************************************
 *
 *  testmpi.c
 *
 *  A simple 'hello world' program to heck the allocation of (GPU)
 *  devices per MPI task.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <mpi.h>

#include "target.h"

int main(int argc, char ** argv) {

  int rank;
  int sz;

  int ndevice;
  int id;
  
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &sz);

  tdpGetDeviceCount(&ndevice);
  tdpGetDevice(&id);

  printf("Rank %d (of %d) has %d devices id %d\n", rank, sz, ndevice, id);

  /* One GPU per MPI rank mandated */
  if (sz == ndevice) {
    struct tdpDeviceProp prop;
    tdpAssert(tdpSetDevice(rank));
    tdpGetDevice(&id);
    printf("Rank %d set device id %d\n", rank, id);
    tdpGetDeviceProperties(&prop, id);
  }
  
  MPI_Finalize();
  
  return 0;
}
