/*****************************************************************************
 *
 *  main.c
 *
 *  Main driver code. See ludwig.c for details of timestepping etc.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
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

  char inputfile[FILENAME_MAX] = "input";


  MPI_Init(&argc, &argv);
#ifdef PETSC
  PetscInitialize(&argc, &argv, (char*) 0, NULL); 
#endif 
  if (argc > 1) sprintf(inputfile, "%s", argv[1]);

  ludwig_run(inputfile);

#ifdef PETSC
  PetscFinalize();
#endif
  MPI_Finalize();

  return 0;
}
