/*****************************************************************************
 *
 *  control.c
 *
 *  Unit test control object.
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unit_control.h"

struct control_s {
  int sz;                  /* Size of control communicator */
  int quiet;               /* Not verbose */
  int root;                /* Root rank in communicator */
  int notroot;             /* ...and not root rank */
  int status;              /* current test status flag */
  int line;                /* current line number of test */
  char file[BUFSIZ];       /* file name string */
  char func[BUFSIZ];       /* function name string */
  MPI_Comm comm;           /* Test communicator (expect MPI_COMM_WORLD) */
};

/*****************************************************************************
 *
 *  control_create
 *
 *  Create the control object under the given communicator.
 *
 *****************************************************************************/

int control_create(MPI_Comm comm, control_t ** pc) throws(MPIException) {

  int rank;
  control_t * c = NULL;
  e4c_mpi_t e;

  e4c_mpi_init(e, comm);

  try {
    c = (control_t *) calloc(1, sizeof(control_t));
    if (c == NULL) throw(NullPointerException, "");

    c->comm = comm;
    c->quiet = CONTROL_QUIET;

    MPI_Comm_size(c->comm, &c->sz);
    MPI_Comm_rank(c->comm, &rank);
    if (rank == 0) c->root = 1;
    if (rank > 0) c->notroot = 1;

    *pc = c;
  }
  catch (NullPointerException) {
    printf("calloc(1, sizeof(control_t)) failed\n");
    e4c_mpi_exception(e);
  }
  finally {
    if (e4c_mpi_allreduce(e)) throw(MPINullPointerException, "");
  }

  return 0;
}

/*****************************************************************************
 *
 *  control_free
 *
 *****************************************************************************/

int control_free(control_t * ctrl) {

  if (ctrl) free(ctrl);

  return 0;
}

/*****************************************************************************
 *
 *  control_comm
 *
 *****************************************************************************/

int control_comm(control_t * ctrl, MPI_Comm * comm) {

  assert(ctrl);

  *comm = ctrl->comm;

  return 0;
}

/*****************************************************************************
 *
 *  control_test
 *
 *****************************************************************************/

int control_test(control_t * ctrl, const char * file, const char * func) {

  assert(ctrl);

  strncpy(ctrl->file, file, BUFSIZ);
  strncpy(ctrl->func, func, BUFSIZ);
  ctrl->status = CONTROL_PASS;
  MPI_Barrier(ctrl->comm);

  control_verb(ctrl, "START %s %s()\n", file, func);

  return 0;
}

/*****************************************************************************
 *
 *  control_allfail
 *
 *  Return global concensus showing CONTROL_FAIL
 *
 *****************************************************************************/

int control_allfail(control_t * ctrl) {

  int ifail = 0;
  int ifail_local;

  ifail_local = (ctrl->status == CONTROL_FAIL);

  MPI_Allreduce(&ifail_local, &ifail, 1, MPI_INT, MPI_LOR, ctrl->comm);

  return ifail;
}

/*****************************************************************************
 *
 *  control_report
 *
 *  Collective test of pass/fail and report
 *
 *****************************************************************************/

int control_report(control_t * ctrl) {

  int nfail = 0;
  int ifail_local;
  int rank;
  int min_rank_local = INT_MAX;
  int min_rank;

  assert(ctrl);

  MPI_Comm_rank(ctrl->comm, &rank);

  /* Work out if anyone has failed and,if so, how many ranks
   * at which point */

  ifail_local = (ctrl->status == CONTROL_FAIL);
  if (ifail_local) min_rank_local = rank;

  MPI_Allreduce(&ifail_local, &nfail, 1, MPI_INT, MPI_SUM, ctrl->comm);
  MPI_Reduce(&min_rank_local, &min_rank, 1, MPI_INT, MPI_MIN, 0, ctrl->comm);

  if (nfail == 0 && ctrl->root) {
    printf("PASS      %s %s\n", ctrl->file, ctrl->func);
  }

  if (nfail == 1 && ctrl->status == CONTROL_FAIL) {
    printf("     FAIL %s %s at line %d [control rank %d]\n",
	   ctrl->file, ctrl->func, ctrl->line, rank);
  }

  if (nfail > 1 && ctrl->root) {
    printf("     FAIL %s %s at line %d [control rank %d (%d ranks)]\n",
	   ctrl->file, ctrl->func, ctrl->line, min_rank, nfail);
  }

  return 0;
}

/*****************************************************************************
 *
 *  control_option_set
 *
 *****************************************************************************/

int control_option_set(control_t * ctrl, control_enum_t key) {

  assert(ctrl);

  if (key == CONTROL_QUIET) ctrl->quiet = CONTROL_QUIET;
  if (key == CONTROL_VERBOSE) ctrl->quiet = CONTROL_VERBOSE;
  if (key == CONTROL_PASS) ctrl->status = CONTROL_PASS;
  if (key == CONTROL_FAIL) ctrl->status = CONTROL_FAIL;

  return 0;
}

/*****************************************************************************
 *
 *  control_verb
 *
 *  Verbose report to stdout at MPI root
 *
 *****************************************************************************/

int control_verb(control_t * ctrl, const char * fmt, ...) {

  va_list args;

  assert(ctrl);
  if (ctrl->notroot || ctrl->quiet == CONTROL_QUIET) return 0;

  fprintf(stdout, "          ");

  va_start(args, fmt);
  vfprintf(stdout, fmt, args);
  va_end(args);

  return 0;
}

/*****************************************************************************
 *
 *  control_fail_report
 *
 *  No synchronisation here else deadlock.
 *
 *****************************************************************************/

int control_fail_line_number_set(control_t * ctrl, int line_number) {

  assert(ctrl);
  ctrl->line = line_number;

  return 0;
}
