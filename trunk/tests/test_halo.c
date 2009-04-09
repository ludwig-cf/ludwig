/*****************************************************************************
 *
 *  test_halo.c
 *
 *  This is a more rigourous test of the halo swap code for the
 *  distributions than appears in test model.
 *
 *  $Id: test_halo.c,v 1.7 2009-04-09 14:54:10 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2007)
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "tests.h"
#include "control.h"

static void test_halo_null(void);
static void test_halo(const int dim);
static int on_corner(int x, int y, int z, int mx, int my, int mz);

int* xfwd;
int* xbwd;
int* yfwd;
int* ybwd;
int* zfwd;
int* zbwd;

int main(int argc, char ** argv) {

  int i,j,k;
  pe_init(argc, argv);

  info("Checking distribution halo swaps...\n\n");

  init_control();
  coords_init();
  init_site();

  xfwd = calloc(NVEL, sizeof(int));
  xbwd = calloc(NVEL, sizeof(int));
  yfwd = calloc(NVEL, sizeof(int));
  ybwd = calloc(NVEL, sizeof(int));
  zfwd = calloc(NVEL, sizeof(int));
  zbwd = calloc(NVEL, sizeof(int));
    
  for (i = 0; i < CVXBLOCK; i++) {
    for (k = 0; k < xblocklen_cv[i]; k++) {
      xfwd[xdisp_fwd_cv[i]+k] = 1;
      xbwd[xdisp_bwd_cv[i]+k] = 1;
    }
  }

  for (i = 0; i < CVYBLOCK; i++) {
    for (k = 0; k < yblocklen_cv[i]; k++) {
      yfwd[ydisp_fwd_cv[i]+k] = 1;
      ybwd[ydisp_bwd_cv[i]+k] = 1;
    }
  }

  for (i = 0; i < CVZBLOCK; i++) {
    for (k = 0; k < zblocklen_cv[i]; k++) {
      zfwd[zdisp_fwd_cv[i]+k] = 1;
      zbwd[zdisp_bwd_cv[i]+k] = 1;
    }
  }

  info("The halo width nhalo_ = %d\n", nhalo_);
  info("Test for null leakage...\n");

  distribution_halo_set_complete();
  info("Full halo...");
  test_halo_null();
  info("ok\n");

  info("Reduced halo...");
  distribution_halo_set_reduced();
  test_halo_null();
  info("ok\n");


  info("Testing x-direction swap...\n");

  distribution_halo_set_complete();
  info("Full halo...");
  test_halo(X);
  info("ok\n");

  distribution_halo_set_reduced();
  info("Reduced halo...");
  test_halo(X);
  info("ok\n");


  info("Testing y-direction swap...\n");

  distribution_halo_set_complete();
  info("Full halo...");
  test_halo(Y);
  info("ok\n");

  distribution_halo_set_reduced();
  info("Reduced halo...");
  test_halo(Y);
  info("ok\n");


  info("Testing z-direction swap...\n");

  distribution_halo_set_complete();
  info("Full halo...");
  test_halo(Z);
  info("ok\n");

  distribution_halo_set_reduced();
  info("Reduced halo...");
  test_halo(Z);
  info("ok\n");

  
  finish_site();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_halo_null
 *
 *  Null halo test. Make sure no halo information appears in the
 *  domain proper.
 *
 *****************************************************************************/

void test_halo_null() {

  int n_local[3], n[3];
  int index, p;
  double f_actual, g_actual;
  int nextra = nhalo_ - 1;

  get_N_local(n_local);

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);

  /* Set entire distribution (all sites including halos) to 1.0 */

  for (n[X] = 1 - nextra; n[X] <= n_local[X] + nextra; n[X]++) {
    for (n[Y] = 1 - nextra; n[Y] <= n_local[Y] + nextra; n[Y]++) {
      for (n[Z] = 1 - nextra; n[Z] <= n_local[Z] + nextra; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, 1.0);
	  set_g_at_site(index, p, 1.0);
	}

      }
    }
  }

  /* Zero interior */

  for (n[X] = 1; n[X] <= n_local[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= n_local[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= n_local[Z]; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, 0.0);
	  set_g_at_site(index, p, 0.0);
	}

      }
    }
  }

  /* Swap */

  halo_site();

  /* Check everywhere in the interior still zero */

  for (n[X] = 1; n[X] <= n_local[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= n_local[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= n_local[Z]; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  f_actual = get_f_at_site(index, p);
	  g_actual = get_g_at_site(index, p);

	  /* everything should still be zero inside the lattice */
	  test_assert(fabs(f_actual - 0.0) < TEST_DOUBLE_TOLERANCE);
	  test_assert(fabs(g_actual - 0.0) < TEST_DOUBLE_TOLERANCE);
	}

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_halo
 *
 *  Test the halo swap for the distributions for coordinate direction dim.
 *
 *****************************************************************************/

void test_halo(int dim) {

  int n_local[3], n[3];
  int offset[3];
  int nextra = nhalo_;
  int index, p, d;
  int * fwd;
  int * bwd;

  double f_expect, f_actual;

  test_assert(dim == X || dim == Y || dim == Z);

  if(dim == X) {
    fwd = xfwd;
    bwd = xbwd;
  }
  if(dim == Y) {
    fwd = yfwd;
    bwd = ybwd;
  }
  if(dim == Z) {
    fwd = zfwd;
    bwd = zbwd;
  }

  get_N_local(n_local);
  get_N_offset(offset);

  /* Zero entire distribution (all sites including halos) */

  for (n[X] = 1 - nextra; n[X] <= n_local[X] + nextra; n[X]++) {
    for (n[Y] = 1 - nextra; n[Y] <= n_local[Y] + nextra; n[Y]++) {
      for (n[Z] = 1 - nextra; n[Z] <= n_local[Z] + nextra; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, -1.0);
	  set_g_at_site(index, p, -1.0);
	}

      }
    }
  }

  /* Set the interior sites to get swapped with a value related to
   * absolute position */

  for (n[X] = 1; n[X] <= n_local[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= n_local[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= n_local[Z]; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	if (n[X] <= nhalo_ || n[X] > n_local[X] - nhalo_ ||
	    n[Y] <= nhalo_ || n[Y] > n_local[Y] - nhalo_ ||
	    n[Z] <= nhalo_ || n[Z] > n_local[Z] - nhalo_) {

	  for (p = 0; p < NVEL; p++) {
	    set_f_at_site(index, p, offset[dim] + n[dim]);
	    set_g_at_site(index, p, offset[dim] + n[dim]);
	  }
	}

      }
    }
  }

  halo_site();

  /* Check the results (all sites for distribution halo).
   * The halo regions should contain a copy of the above, while the
   * interior sites are unchanged */

  /* Note the distribution halo swaps are always width 1, irrespective
   * of nhalo_ */

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	for (d = 0; d < 3; d++) {

	  /* 'Left' side */
	  if (dim == d && n[d] == 0) {

	    f_expect = offset[dim];
	    if (cart_coords(dim) == 0) f_expect = L(dim);

	    for (p = 0; p < NVEL; p++) {
	      f_actual = get_f_at_site(index, p);
	      test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	    }
	  }

	  /* 'Right' side */
	  if (dim == d && n[d] == n_local[d] + 1) {

	    f_expect = offset[dim] + n_local[dim] + 1.0;
	    if (cart_coords(dim) == cart_size(dim) - 1) f_expect = 1.0;

	    for (p = 0; p < NVEL; p++) {
	      f_actual = get_f_at_site(index, p);
	      test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	    }
	  }
	}
	/* Next site */
      }
    }
  }

  return;
}


/*************************************
 *
 * Returns 0(false) if on a corner,
 *         1(true)  otherwise.
 *
 *************************************/

static int on_corner(int x, int y, int z, int mx, int my, int mz) {

  int iscorner = 0;

  /* on the axes */
  if (fabs(x) + fabs(y) == 0 || fabs(x) + fabs(z) == 0 ||
      fabs(y) + fabs(z) == 0 ) {
    iscorner = 1;
  }

  /* opposite corners from axes */

  if ((x == mx && y == my) || (x == mx && z == mz) || (y == my && z == mz)) {
      iscorner = 1;
  }
  
  if ((x == 0 && y == my) || (x == 0 && z == mz) || (y == 0 && x == mx) ||
      (y == 0 && z == mz) || (z == 0 && x == mx) || (z == 0 && y == my)) {
    iscorner = 1;
  }

  return iscorner;
}
