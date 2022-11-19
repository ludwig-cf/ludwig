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

int process_command_line(const char * arg, char * filename, size_t bufsz);

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
  if (argc > 1) snprintf(inputfile, FILENAME_MAX, "%s", argv[1]);
#endif 

  if (argc == 1) {
    ludwig_run("input");
  }
  else if (argc > 1) {
    char filename[BUFSIZ/2] = {0};
    int ifail = process_command_line(argv[1], filename, BUFSIZ/2);
    if (ifail == 0) {
      char buf[BUFSIZ] = "./";
      char * f = buf;
      strncat(f+2, filename, BUFSIZ-3);
      ludwig_run(f);
    }
    else {
      printf("Input file name: %s\n"
	     "Please use a posix file name with only alphanumeric\n"
	     "characters or _ or - or .\n", argv[1]);
    }
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

int process_command_line(const char * arg, char * filename, size_t bufsz) {

  int ifail = -1;

  /* The first character should be alphabetical */ 

  if (isalpha(arg[0])) {
    size_t len = strnlen(arg, bufsz-1);
    for (size_t i = 0; i < len; i++) {
      const char c = arg[i];
      filename[i] = '_';
      if (isalnum(c) || c == '_' || c == '-' || c == '.') filename[i] = c;
    }
    filename[len] = '\0';
    ifail = strncmp(arg, filename, bufsz-1);
  }

  return ifail;
}
