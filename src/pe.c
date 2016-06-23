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
 *  (c) 2010-2014 The University of Edinburgh
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "svn.h"
#include "pe.h"

struct pe_s {
  int unquiet;                       /* Print version information etc */
  int mpi_rank;                      /* Rank in dup'd comm */
  int mpi_size;                      /* Size of comm */
  MPI_Comm parent_comm;              /* Reference to parrent entering dup */
  MPI_Comm comm;                     /* Communicator for pe */
  char subdirectory[FILENAME_MAX];
};

static pe_t * pe = NULL;
static int pe_create(MPI_Comm parent);

/*****************************************************************************
 *
 *  pe_create
 *
 *  Single static instance. MPI must be initialised.
 *
 *****************************************************************************/

static int pe_create(MPI_Comm parent) {

  int ifail_local = 0;
  int ifail;

  assert(pe == NULL);

  MPI_Initialized(&ifail);

  if (ifail == 0) {
    printf("Please make sure MPI is initialised!\n");
    exit(0);
  }

  pe = (pe_t *) calloc(1, sizeof(pe_t));
  if (pe == NULL) ifail_local = 1;
  MPI_Allreduce(&ifail_local, &ifail, 1, MPI_INT, MPI_SUM, parent);

  if (ifail != 0) {
    printf("calloc(pe_t) failed\n");
    exit(0);
  }

  pe->parent_comm = parent;
  strcpy(pe->subdirectory, "");

  return 0;
}

/*****************************************************************************
 *
 *  pe_init
 *
 *  Initialise the model. If it's MPI, we choose that all errors
 *  be terminal.
 *
 *****************************************************************************/

void pe_init(void) {

  pe_init_quiet();
  pe->unquiet = 1;

  info("Welcome to Ludwig v%d.%d.%d (%s version running on %d process%s)\n\n",
       LUDWIG_MAJOR_VERSION, LUDWIG_MINOR_VERSION, LUDWIG_PATCH_VERSION,
       (pe->mpi_size > 1) ? "MPI" : "Serial", pe->mpi_size,
       (pe->mpi_size == 1) ? "" : "es");

  if (pe->mpi_rank == 0) {
    printf("The SVN revision details are: %s\n", svn_revision());
    assert(printf("Note assertions via standard C assert() are on.\n\n"));
  }

  return;
}

/*****************************************************************************
 *
 *  pe_init_quiet
 *
 *****************************************************************************/

int pe_init_quiet(void) {

  if (pe == NULL) pe_create(MPI_COMM_WORLD);

  MPI_Comm_dup(pe->parent_comm, &pe->comm);

  MPI_Comm_set_errhandler(pe->comm, MPI_ERRORS_ARE_FATAL);

  MPI_Comm_size(pe->comm, &pe->mpi_size);
  MPI_Comm_rank(pe->comm, &pe->mpi_rank);

  return 0;
}

/*****************************************************************************
 *
 *  pe_finalise
 *
 *  This is the final executable statement.
 *
 *****************************************************************************/

void pe_finalise() {

  assert(pe);

  MPI_Comm_free(&pe->comm);
  if (pe->unquiet) info("Ludwig finished normally.\n");

  free(pe);
  pe = NULL;

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

  assert(pe);

  MPI_Comm_rank(pe->parent_comm, &rank);

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

  assert(pe);
  return pe->comm;
}

/*****************************************************************************
 *
 *  pe_rank, pe_size
 *
 *  "Getter" functions.
 *
 *****************************************************************************/

int pe_rank() {
  assert(pe);
  return pe->mpi_rank;
}

int pe_size() {
  assert(pe);
  return pe->mpi_size;
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

  assert(pe);

  if (pe->mpi_rank == 0) {
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

  assert(pe);

  printf("[%d] ", pe->mpi_rank);

  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  /* Considered a successful exit (code 0). */

  MPI_Abort(pe->comm, 0);

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

  assert(pe);

  printf("[%d] ", pe->mpi_rank);

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

  assert(pe);
  pe->parent_comm = parent;

  return;
}

/*****************************************************************************
 *
 *  pe_subdirectory_set
 *
 *****************************************************************************/

void pe_subdirectory_set(const char * name) {

  assert(pe);
  if (name != NULL) sprintf(pe->subdirectory, "%s/", name);

  return;
}

/*****************************************************************************
 *
 *  pe_subdirectory
 *
 *****************************************************************************/

void pe_subdirectory(char * name) {

  assert(pe);
  assert(name);

  sprintf(name, "%s", pe->subdirectory);

  return;
}
