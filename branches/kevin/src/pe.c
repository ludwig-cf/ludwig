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
  int nref;                          /* Reference count */
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
  pe->nref = 1;
  strcpy(pe->subdirectory, "");

  return 0;
}

/*****************************************************************************
 *
 *  pe_retain
 *
 *  Increment reference count by one.
 *
 *****************************************************************************/

int pe_retain(pe_t * peref) {

  assert(peref);

  peref->nref += 1;

  return 0;
}

/*****************************************************************************
 *
 *  pe_free
 *
 *  Release a reference; if that's the last reference, close down.
 *
 *****************************************************************************/

int pe_free(pe_t * peref) {

  assert(peref);

  peref->nref -= 1;

  if (peref->nref <= 0) {
    MPI_Comm_free(&pe->comm);
    if (pe->unquiet) info("Ludwig finished normally.\n");
    free(pe);
    pe = NULL;
  }

  return 0;
}

/*****************************************************************************
 *
 *  pe_set
 *
 *****************************************************************************/

int pe_set(pe_t * pe, pe_enum_t option) {

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

int pe_mpi_comm(pe_t * peref, MPI_Comm * comm) {

  assert(peref);
  assert(comm);

  *comm = pe->comm;

  return 0;
}

/*****************************************************************************
 *
 *  pe_mpi_rank
 *
 *****************************************************************************/

int pe_mpi_rank(pe_t * peref) {

  assert(peref);

  return peref->mpi_rank;
}

/*****************************************************************************
 *
 *  pe_mpi_size
 *
 *****************************************************************************/

int pe_mpi_size(pe_t * peref) {

  assert(peref);

  return peref->mpi_size;
}


/*****************************************************************************
 *
 *  pe_commit
 *
 *  Just the welcome message.
 *
 *  The message. Don't push me 'cos I'm close to the edge...
 *
 *****************************************************************************/

int pe_commit(pe_t * pe) {

  assert(pe);

  info("Welcome to Ludwig v%d.%d.%d (%s version running on %d process%s)\n\n",
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
 *  pe_create_parent
 *
 *****************************************************************************/

int pe_create_parent(MPI_Comm parent, pe_t ** peref) {

  int ifail = 0;

  assert(pe == NULL);

  ifail = pe_create(parent);

  MPI_Comm_dup(pe->parent_comm, &pe->comm);

  MPI_Errhandler_set(pe->comm, MPI_ERRORS_ARE_FATAL);

  MPI_Comm_size(pe->comm, &pe->mpi_size);
  MPI_Comm_rank(pe->comm, &pe->mpi_rank);

  if (ifail == 0) *peref = pe;

  return ifail;
}

/*****************************************************************************
 *
 *  pe_subdirectory_set
 *
 *****************************************************************************/

int pe_subdirectory_set(pe_t * pe, const char * name) {

  assert(pe);
  if (name != NULL) sprintf(pe->subdirectory, "%s/", name);

  return 0;
}

/*****************************************************************************
 *
 *  pe_subdirectory
 *
 *****************************************************************************/

int pe_subdirectory(pe_t * pe, char * name) {

  assert(pe);
  assert(name);

  sprintf(name, "%s", pe->subdirectory);

  return 0;
}
