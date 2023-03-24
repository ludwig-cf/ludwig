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
#include "util_petsc.h"

/*****************************************************************************
 *
 *  main
 *
 *  The Petsc initialisation/finalisation is a facade if there's no
 *  actual Petsc in the build.
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  char inputfile[FILENAME_MAX] = "input";
  int provided = MPI_THREAD_SINGLE;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  PetscInitialize(&argc, &argv, (char*) 0, NULL); 

  if (argc > 1) snprintf(inputfile, FILENAME_MAX, "%s", argv[1]);

  ludwig_run(inputfile);

  PetscFinalize();
  MPI_Finalize();

  return 0;
}
