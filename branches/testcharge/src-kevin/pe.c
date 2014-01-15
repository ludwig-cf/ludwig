/*****************************************************************************
 *
 *  pe.c
 *
 *  The parallel environment.
 *
 *  This is responsible for initialisation and finalisation of
 *  the parallel environment. In serial, the MPI stub library is
 *  required.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "svn.h"
#include "pe.h"

static int pe_world_rank;
static int pe_world_size;
static MPI_Comm pe_parent_comm_ = MPI_COMM_WORLD;
static MPI_Comm pe_comm_;
static char subdirectory_[FILENAME_MAX] = "";

/*****************************************************************************
 *
 *  pe_init
 *
 *  Initialise the model. If it's MPI, we choose that all errors
 *  be terminal.
 *
 *****************************************************************************/

void pe_init(void) {

  int ifail;

  MPI_Initialized(&ifail);

  if (ifail == 0) {
    printf("Please make sure MPI is initialised!\n");
    exit(0);
  }

  MPI_Comm_dup(pe_parent_comm_, &pe_comm_);

  MPI_Errhandler_set(pe_comm_, MPI_ERRORS_ARE_FATAL);

  MPI_Comm_size(pe_comm_, &pe_world_size);
  MPI_Comm_rank(pe_comm_, &pe_world_rank);

  info("Welcome to Ludwig (%s version running on %d process%s)\n\n",
       (pe_world_size > 1) ? "MPI" : "Serial", pe_world_size,
       (pe_world_size == 1) ? "" : "es");

  if (pe_world_rank == 0) {
    printf("The SVN revision details are: %s\n", svn_revision());
    assert(printf("Note assertions via standard C assert() are on.\n\n"));
  }

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

  MPI_Comm_free(&pe_comm_);
  info("Ludwig finished normally.\n");

  return;
}

/*****************************************************************************
 *
 *  pe_redirect_stdout
 *
 *****************************************************************************/

void pe_redirect_stdout(const char * filename) {

  int rank;
  FILE * stream;

  MPI_Comm_rank(pe_parent_comm_, &rank);

  if (rank == 0) {
    printf("Redirecting stdout to file %s\n", filename);
  }

  stream = freopen(filename, "w", stdout);
  if (stream == NULL) {
    printf("[%d] ffreopen(%s) failed\n", rank, filename);
    fatal("Stop.\n");
  }

  return;
}

/*****************************************************************************
 *
 *  pe_comm
 *
 *****************************************************************************/

MPI_Comm pe_comm(void) {

  return pe_comm_;
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

  MPI_Abort(pe_comm_, 0);

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

  printf("[%d] ", pe_world_rank);

  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  return;
}

/*****************************************************************************
 *
 *  pe_parent_comm_set
 *
 *****************************************************************************/

void pe_parent_comm_set(MPI_Comm parent) {

  pe_parent_comm_ = parent;
  return;
}

/*****************************************************************************
 *
 *  pe_subdirectory_set
 *
 *****************************************************************************/

void pe_subdirectory_set(const char * name) {

  if (name != NULL) sprintf(subdirectory_, "%s/", name);

  return;
}

/*****************************************************************************
 *
 *  pe_subdirectory
 *
 *****************************************************************************/

void pe_subdirectory(char * name) {

  sprintf(name, "%s", subdirectory_);

  return;
}
