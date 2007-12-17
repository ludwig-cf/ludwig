/*****************************************************************************
 *
 *  pe.c
 *
 *  The parallel environment.
 *
 *  This is responsible for initialisation and finalisation of
 *  the parallel environment (or a logically consistent picture
 *  in serial). Functions to get the current rank and size in
 *  MPI_COMM_WORLD are provided.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "pe.h"

static int pe_world_rank = 0;
static int pe_world_size = 1;

/*****************************************************************************
 *
 *  pe_init
 *
 *  Initialise the model. If it's MPI, we choose that all errors
 *  be terminal.
 *
 *****************************************************************************/

void pe_init(int argc, char ** argv) {

#ifdef _MPI_
  MPI_Init(&argc, &argv);

  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

  MPI_Comm_size(MPI_COMM_WORLD, &pe_world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &pe_world_rank);
#endif

  info("Welcome to Ludwig (%s version running on %d process%s)\n\n",
       (pe_world_size > 1) ? "MPI" : "Serial",
       pe_world_size,
       (pe_world_size == 1) ? "" : "es");

  return;
}

/*****************************************************************************
 *
 *  pe_finalise
 *
 *  This is the final executable statement.
 *
 *****************************************************************************/

void pe_finalise() {

  info("Ludwig finished normally.\n");

#ifdef _MPI_
  MPI_Finalize();
#endif

  return;
}

/*****************************************************************************
 *
 *  pe_rank, pe_size
 *
 *  "Getter" functions.
 *
 *****************************************************************************/

int pe_rank() {
  return pe_world_rank;
}

int pe_size() {
  return pe_world_size;
}

/*****************************************************************************
 *
 *  info
 *
 *  Print arguments on process 0 only (important stuff!).
 *
 *****************************************************************************/

void info(const char * fmt, ...) {

  va_list args;

  if (pe_world_rank == 0) {
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
  }

  return;
}

/*****************************************************************************
 *
 *  fatal
 *
 *  Terminate the program with a message from the offending process.
 *
 *****************************************************************************/

void fatal(const char * fmt, ...) {

  va_list args;

  printf("[%d] ", pe_world_rank);
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  /* Considered a successful exit (code 0). */
#ifdef _MPI_
  MPI_Abort(MPI_COMM_WORLD, 0);
#else
  exit(0);
#endif

  return;
}

/*****************************************************************************
 *
 *  verbose
 *
 *  Always prints a message.
 *
 *****************************************************************************/

void verbose(const char * fmt, ...) {

  va_list args;

#ifdef _MPI_
  printf("[%d] ", pe_world_rank);
#endif
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  return;
}

/*****************************************************************************
 *
 *  imin, imax, dmin, dmax
 *
 *  minimax functions
 *
 *****************************************************************************/

int imin(const int i, const int j) {
  return ((i < j) ? i : j);
}

int imax(const int i, const int j) {
  return ((i > j) ? i : j);
}

double dmin(const double a, const double b) {
  return ((a < b) ? a : b);
}

double dmax(const double a, const double b) {
  return ((a > b) ? a : b);
}
