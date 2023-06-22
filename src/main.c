/*****************************************************************************
 *
 *  main.c
 *
 *  Main driver code. See ludwig.c for details of timestepping etc.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011-2023 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>

#include "pe.h"
#include "ludwig.h"
#ifdef PETSC
  #include "petscksp.h"
#endif

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  const char * inputfile = "input";
  int provided = MPI_THREAD_SINGLE;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#ifdef PETSC
  PetscInitialize(&argc, &argv, (char*) 0, NULL); 
#endif 

  if (argc == 1) {
    ludwig_run(inputfile);
  }
  else {
    /* No command line arguments please */
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      printf("Command line arguments are now disabled.\n");
      printf("In particular, the input file must be called \"input\"\n");
      printf("and be in the current working directory.\n");
    }
  }

#ifdef PETSC
  PetscFinalize();
#endif
  MPI_Finalize();

  return 0;
}
