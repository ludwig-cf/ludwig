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

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "ludwig.h"
#ifdef PETSC
  #include "petscksp.h"
#endif

int process_command_line(const char * arg);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int provided = MPI_THREAD_SINGLE;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#ifdef PETSC
  PetscInitialize(&argc, &argv, (char*) 0, NULL); 
#endif 

  if (argc == 1) {
    ludwig_run("input");
  }
  else if (argc > 1 && process_command_line(argv[1]) == 0) {
    ludwig_run(argv[1]);
  }

#ifdef PETSC
  PetscFinalize();
#endif
  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  process_command_line
 *
 *  Require a posix portable filename, with the additional condition
 *  that the first character is alphabetical.
 *
 *  A valid filename will return zero.
 *
 *****************************************************************************/

int process_command_line(const char * arg) {

  int ifail = -1;

  /* The first character should be alphabetical */ 

  if (isalpha(arg[0])) {
    for (size_t i = 0; i < strnlen(arg, FILENAME_MAX); i++) {
      const char c = arg[i];
      ifail = -1;
      if (isalnum(c) || c == '_' || c == '-' || c == '.') ifail = 0;
    }
  }

  if (ifail != 0) {
    printf("Input file name: %s\n"
	   "Please use a posix file name with only alphanumeric\n"
	   "characters or _ or - or .\n", arg);
  }

  return ifail;
}
