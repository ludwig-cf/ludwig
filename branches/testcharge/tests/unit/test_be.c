/*****************************************************************************
 *
 *  test_be.c
 *
 *  Test for Beris Edwards solver.
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "mpi.h"
#include "blue_phase_beris_edwards.h"

static int do_test_be_tmatrix(void);


/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);

  do_test_be_tmatrix();

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_be_matrix
 *
 *  Check the noise basis t^i_ab t^j_ab = \delta_ij
 *
 *****************************************************************************/

static int do_test_be_tmatrix(void) {

  int i, j, ia, ib;
  double sum, dij;
  double t[3][3][NQAB];

  blue_phase_be_tmatrix_set(t);

  for (i = 0; i < NQAB; i++) {
    for (j = 0; j < NQAB; j++) {

      sum = 0.0;
      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  sum += t[ia][ib][i]*t[ia][ib][j];
	}
      }
      dij = (i == j);
      assert(fabs(sum - dij) <= DBL_EPSILON);
    }
  }

  return 0;
}
