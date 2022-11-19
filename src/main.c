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
 *  (c) 2011-2022 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "ludwig.h"
#ifdef PETSC
  #include "petscksp.h"
#endif

int process_command_line(const char * arg, char * filename, size_t bufsz);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  char inputfile[FILENAME_MAX] = "input";
  int provided = MPI_THREAD_SINGLE;


  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#ifdef PETSC
  PetscInitialize(&argc, &argv, (char*) 0, NULL); 
  if (argc > 1) snprintf(inputfile, FILENAME_MAX, "%s", argv[1]);
#endif 

  if (argc > 1) {
    int ifail = process_command_line(argv[1], inputfile, FILENAME_MAX);
    if (ifail != 0) {
      printf("Input file name %s is not valid\n", argv[1]);
      exit(-1);
    }
  }

  ludwig_run(inputfile);

#ifdef PETSC
  PetscFinalize();
#endif
  MPI_Finalize();

  return 0;
}

#include <ctype.h>
#include <string.h>

int process_command_line(const char * arg, char * filename, size_t bufsz) {

  int ifail = 0;

  /* The first character should be alphabetical */ 

  if (isalpha(arg[0])) {
    int ndot = 0;
    size_t len = strnlen(arg, bufsz-1);
    for (size_t i = 0; i < len; i++) {
      const char c = arg[i];
      if (c == '.') ndot += 1;
      filename[i] = '_';
      if (isalnum(c) || c == '_' || c == '-' || c == '.') filename[i] = c;
    }
    filename[len] = '\0';
    ifail = strncmp(arg, filename, bufsz-1);
  }

  return ifail;
}
