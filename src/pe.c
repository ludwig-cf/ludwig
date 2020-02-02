/*****************************************************************************
 *
 *  pe.c
 *
 *  The parallel environment.
 *
 *  This is responsible for initialisation and finalisation of
 *  information on the parallel environment (MPI, thread model).
 *
 *  Prints basic information to a root process, or verbosely.
 *
 *  In serial, the MPI stub library is required.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "pe.h"

struct pe_s {
  int unquiet;                       /* Print version information etc */
  int mpi_rank;                      /* Rank in dup'd comm */
  int mpi_size;                      /* Size of comm */
  int nref;                          /* Retained reference count */
  MPI_Comm parent_comm;              /* Reference to parent communicator */
  MPI_Comm comm;                     /* Communicator for pe itself */
  char subdirectory[FILENAME_MAX];
};

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

  assert(pe);
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

  *ppe = pe;
  
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
    assert(printf("Note assertions via standard C assert() are on.\n\n"));
    tdpThreadModelInfo(stdout);
    printf("\n");
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
 *  Always prints a message (prefixed by MPI rank).
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
