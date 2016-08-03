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
 *  A static reference is retained to provide access deep in the
 *  call tree via pe_ref(). This should ultimately be removed.
 *
 *  $Id$
 *
 *  (c) 2010-2016 The University of Edinburgh
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
  int nref;                          /* Retained reference count */
  MPI_Comm parent_comm;              /* Reference to parrent entering dup */
  MPI_Comm comm;                     /* Communicator for pe */
  char subdirectory[FILENAME_MAX];
};

static pe_t * pe_static = NULL;

/*****************************************************************************
 *
 *  pe_create
 *
 *  Create a duplicate of the parent. Initialise error handling.
 *
 *****************************************************************************/

__host__ int pe_create(MPI_Comm parent, pe_enum_t flag, pe_t ** ppe) {

  int ifail_local = 0;
  int ifail;
  pe_t * pe = NULL;

  assert(ppe);

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

  pe->unquiet = 0; /* Quiet */
  pe->parent_comm = parent;
  pe->nref = 1;
  strcpy(pe->subdirectory, "");

  MPI_Comm_dup(parent, &pe->comm);
  MPI_Comm_set_errhandler(pe->comm, MPI_ERRORS_ARE_FATAL);

  MPI_Comm_size(pe->comm, &pe->mpi_size);
  MPI_Comm_rank(pe->comm, &pe->mpi_rank);

  if (flag == PE_VERBOSE) {
    pe->unquiet = 1;
    pe_message(pe);
  }

  pe_static = pe;
  *ppe = pe;
  
  return 0;
}

/*****************************************************************************
 *
 *  pe_ref
 *
 *****************************************************************************/

__host__ int pe_ref(pe_t ** ppe) {

  assert(pe_static);

  *ppe = pe_static;

  return 0;
}

/*****************************************************************************
 *
 *  pe_retain
 *
 *  Increment reference count by one.
 *
 *****************************************************************************/

__host__ int pe_retain(pe_t * pe) {

  assert(pe);

  pe->nref += 1;

  return 0;
}

/*****************************************************************************
 *
 *  pe_free
 *
 *  Release a reference; if that's the last reference, close down.
 *
 *****************************************************************************/

__host__ int pe_free(pe_t * pe) {

  assert(pe);

  pe->nref -= 1;

  if (pe->nref <= 0) {
    MPI_Comm_free(&pe->comm);
    if (pe->unquiet) pe_info(pe, "Ludwig finished normally.\n");
    free(pe);
    pe = NULL;
    pe_static = NULL;
  }

  return 0;
}

/*****************************************************************************
 *
 *  pe_message
 *
 *  Banner message shown at start of execution.
 *
 *****************************************************************************/

__host__ int pe_message(pe_t * pe) {

  assert(pe);

  pe_info(pe,
       "Welcome to Ludwig v%d.%d.%d (%s version running on %d process%s)\n\n",
       LUDWIG_MAJOR_VERSION, LUDWIG_MINOR_VERSION, LUDWIG_PATCH_VERSION,
       (pe->mpi_size > 1) ? "MPI" : "Serial", pe->mpi_size,
       (pe->mpi_size == 1) ? "" : "es");

  if (pe->mpi_rank == 0) {
    printf("The SVN revision details are: %s\n", svn_revision());
    assert(printf("Note assertions via standard C assert() are on.\n\n"));
  }

  return 0;
}

/*****************************************************************************
 *
 *  pe_info
 *
 *  Print arguments on process 0 only (important stuff!).
 *
 *****************************************************************************/

__host__ int pe_info(pe_t * pe, const char * fmt, ...) {

  va_list args;

  assert(pe);

  if (pe->mpi_rank == 0) {
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pe_fatal
 *
 *  Terminate the program with a message from the offending process.
 *
 *****************************************************************************/

__host__ int pe_fatal(pe_t * pe, const char * fmt, ...) {

  va_list args;

  assert(pe);

  printf("[%d] ", pe->mpi_rank);

  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  /* Considered a successful exit (code 0). */

  MPI_Abort(pe->comm, 0);

  return 0;
}

/*****************************************************************************
 *
 *  pe_verbose
 *
 *  Always prints a message.
 *
 *****************************************************************************/

__host__ int pe_verbose(pe_t * pe, const char * fmt, ...) {

  va_list args;

  assert(pe);

  printf("[%d] ", pe->mpi_rank);

  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  return 0;
}

/*****************************************************************************
 *
 *  pe_subdirectory_set
 *
 *****************************************************************************/

__host__ int pe_subdirectory_set(pe_t * pe, const char * name) {

  assert(pe);
  if (name != NULL) sprintf(pe->subdirectory, "%s/", name);

  return 0;
}

/*****************************************************************************
 *
 *  pe_subdirectory
 *
 *****************************************************************************/

__host__ int pe_subdirectory(pe_t * pe, char * name) {

  assert(pe);
  assert(name);

  sprintf(name, "%s", pe->subdirectory);

  return 0;
}

/*****************************************************************************
 *
 *  pe_set
 *
 *****************************************************************************/

__host__ int pe_set(pe_t * pe, pe_enum_t option) {

  assert(pe);

  switch (option) {
  case PE_QUIET:
    pe->unquiet = 0;
    break;
  case PE_VERBOSE:
    pe->unquiet = 1;
    break;
  default:
    assert(0);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pe_mpi_comm
 *
 *****************************************************************************/

__host__ int pe_mpi_comm(pe_t * pe, MPI_Comm * comm) {

  assert(pe);
  assert(comm);

  *comm = pe->comm;

  return 0;
}

/*****************************************************************************
 *
 *  pe_mpi_rank
 *
 *****************************************************************************/

__host__ int pe_mpi_rank(pe_t * pe) {

  assert(pe);

  return pe->mpi_rank;
}

/*****************************************************************************
 *
 *  pe_mpi_size
 *
 *****************************************************************************/

__host__ int pe_mpi_size(pe_t * pe) {

  assert(pe);

  return pe->mpi_size;
}

/*****************************************************************************
 *
 *  pe_comm, pe_rank, pe_size, ...
 *
 *****************************************************************************/

__host__ MPI_Comm pe_comm(void) {
  MPI_Comm comm;
  assert(pe_static);
  pe_mpi_comm(pe_static, &comm);
  return comm;
}

__host__ int pe_rank(void) {
  assert(pe_static);
  return pe_mpi_rank(pe_static);
}

__host__ int pe_size(void) {
  assert(pe_static);
  return pe_mpi_size(pe_static);
}

__host__ void info(const char * fmt, ...) {

  va_list args;

  assert(pe_static);

  if (pe_static->mpi_rank == 0) {
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
  }

  return;
}

__host__ void fatal(const char * fmt, ...) {

  va_list args;

  assert(pe_static);

  printf("[%d] ", pe_static->mpi_rank);

  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  /* Considered a successful exit (code 0). */

  MPI_Abort(pe_static->comm, 0);

  return;
}

__host__ void verbose(const char * fmt, ...) {

  va_list args;

  assert(pe_static);

  printf("[%d] ", pe_static->mpi_rank);

  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  return;
}

