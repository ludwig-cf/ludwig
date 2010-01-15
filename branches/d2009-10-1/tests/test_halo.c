/*****************************************************************************
 *
 *  test_halo.c
 *
 *  This is a more rigourous test of the halo swap code for the
 *  distributions than appears in test model.
 *
 *  $Id: test_halo.c,v 1.8.2.2 2010-01-15 17:10:40 kevin Exp $
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
static void test_halo(const int dim, const int reduced);
static int on_edge(int x, int y, int z, int mx, int my, int mz);

int main(int argc, char ** argv) {

  int i, k;
  pe_init(argc, argv);

  info("Checking distribution halo swaps...\n\n");

  init_control();
  coords_init();
  init_site();

  /* Check the reduced halo blocks. */

  info("Checking the reduced halo blocks...");
    
  for (i = 0; i < CVXBLOCK; i++) {
    for (k = 0; k < xblocklen_cv[i]; k++) {
      test_assert(cv[xdisp_fwd_cv[i] + k][X] == +1);
      test_assert(cv[xdisp_bwd_cv[i] + k][X] == -1);
    }
  }

  for (i = 0; i < CVYBLOCK; i++) {
    for (k = 0; k < yblocklen_cv[i]; k++) {
      test_assert(cv[ydisp_fwd_cv[i] + k][Y] == +1);
      test_assert(cv[ydisp_bwd_cv[i] + k][Y] == -1);
    }
  }

  for (i = 0; i < CVZBLOCK; i++) {
    for (k = 0; k < zblocklen_cv[i]; k++) {
      test_assert(cv[zdisp_fwd_cv[i] + k][Z] == +1);
      test_assert(cv[zdisp_bwd_cv[i] + k][Z] == -1);
    }
  }

  info("ok\n");

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
  test_halo(X, 0);
  info("ok\n");

  distribution_halo_set_reduced();
  info("Reduced halo...");
  test_halo(X, 1);
  info("ok\n");


  info("Testing y-direction swap...\n");

  distribution_halo_set_complete();
  info("Full halo...");
  test_halo(Y, 0);
  info("ok\n");

  distribution_halo_set_reduced();
  info("Reduced halo...");
  test_halo(Y, 1);
  info("ok\n");


  info("Testing z-direction swap...\n");

  distribution_halo_set_complete();
  info("Full halo...");
  test_halo(Z, 0);
  info("ok\n");

  distribution_halo_set_reduced();
  info("Reduced halo...");
  test_halo(Z, 1);
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
 *  domain proper. This works for both full and reduced halos.
 *
 *****************************************************************************/

void test_halo_null() {

  int n_local[3], n[3];
  int index, nd, p;
  int ndist;
  double f_actual;
  int nextra = nhalo_ - 1;

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);

  get_N_local(n_local);
  ndist = distribution_ndist();

  /* Set entire distribution (all sites including halos) to 1.0 */

  for (n[X] = 1 - nextra; n[X] <= n_local[X] + nextra; n[X]++) {
    for (n[Y] = 1 - nextra; n[Y] <= n_local[Y] + nextra; n[Y]++) {
      for (n[Z] = 1 - nextra; n[Z] <= n_local[Z] + nextra; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    distribution_f_set(index, p, nd, 1.0);
	  }
	}

      }
    }
  }

  /* Zero interior */

  for (n[X] = 1; n[X] <= n_local[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= n_local[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= n_local[Z]; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    distribution_f_set(index, p, nd, 0.0);
	  }
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

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    f_actual = distribution_f(index, p, nd);

	    /* everything should still be zero inside the lattice */
	    test_assert(fabs(f_actual - 0.0) < TEST_DOUBLE_TOLERANCE);
	  }
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
 *  Note that the reduced halo swaps are only meaningful in
 *  parallel. They will automatically work in serial.
 *
 *****************************************************************************/

void test_halo(int dim, int reduced) {

  int n_local[3], n[3];
  int offset[3];
  int ic, jc, kc;
  int nd, ndist;
  int nextra = nhalo_;
  int index, p, d;

  double f_expect, f_actual;

  test_assert(dim == X || dim == Y || dim == Z);

  get_N_local(n_local);
  get_N_offset(offset);
  ndist = distribution_ndist();

  /* Zero entire distribution (all sites including halos) */

  for (n[X] = 1 - nextra; n[X] <= n_local[X] + nextra; n[X]++) {
    for (n[Y] = 1 - nextra; n[Y] <= n_local[Y] + nextra; n[Y]++) {
      for (n[Z] = 1 - nextra; n[Z] <= n_local[Z] + nextra; n[Z]++) {

	index = get_site_index(n[X], n[Y], n[Z]);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    distribution_f_set(index, p, nd, -1.0);
	  }
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

	  for (nd = 0; nd < ndist; nd++) {
	    for (p = 0; p < NVEL; p++) {
	      distribution_f_set(index, p, nd, 1.0*(offset[dim] + n[dim]));
	    }
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

	for (nd = 0; nd < ndist; nd++) {
	  for (d = 0; d < 3; d++) {

	    /* 'Left' side */
	    if (dim == d && n[d] == 0) {

	      f_expect = offset[dim];
	      if (cart_coords(dim) == 0) f_expect = L(dim);

	      for (p = 0; p < NVEL; p++) {
		f_actual = distribution_f(index, p, nd);
		if (reduced) {
		}
		else {
		  test_assert(fabs(f_actual-f_expect) < TEST_DOUBLE_TOLERANCE);
		}
	      }
	    }

	    /* 'Right' side */
	    if (dim == d && n[d] == n_local[d] + 1) {

	      f_expect = offset[dim] + n_local[dim] + 1.0;
	      if (cart_coords(dim) == cart_size(dim) - 1) f_expect = 1.0;

	      for (p = 0; p < NVEL; p++) {
		if (reduced) {
		}
		else {
		  f_actual = distribution_f(index, p, nd);
		  test_assert(fabs(f_actual-f_expect) < TEST_DOUBLE_TOLERANCE);
		}
	      }
	    }
	  }
	}
	/* Next site */
      }
    }
  }

  /* REDUCED HALO NEEDS WORK*/
  /* The logic required for the edges and corners for general
   * decomposition is really more pain than it is worth. By
   * excluding the edges and corners, a few cases may be
   * missed. The true test is therefore in the propagation
   * (see test_prop.c). */
   
  if (reduced && dim == X && cart_size(X) > 1) {

    for (jc = 0; jc <= n_local[Y] + 1; jc++) {
      for (kc = 0; kc <= n_local[Z] + 1; kc++) {

	/* left hand edge */
	index = get_site_index(0, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    f_actual = distribution_f(index, p, nd);
	    f_expect = -1.0;

	    if (cv[p][X] > 0) {
	      f_expect = offset[X];
	      if (cart_coords(dim) == 0) f_expect = L(X);
	    }

	  }

	  /* right hand edge */
	  ic = n_local[X] + 1;
	  index = get_site_index(ic, jc, kc);

	  for (p = 0; p < NVEL; p++) {
	    f_actual = distribution_f(index, p, nd);
	    f_expect = -1.0;

	    if (cv[p][X] < 0) {
	      f_expect = offset[X] + ic;
	      if (cart_coords(X) == cart_size(X) - 1) f_expect = 1.0;
	    }
	  }
	}
	/* Next site */
      }
    }

    /* Finish x direction */
  }

  /* Y-DIRECTION */

  if (reduced && dim == Y && cart_size(Y) > 1) {

    for (ic = 0; ic <= n_local[X] + 1; ic++) {
      for (kc = 0; kc <= n_local[Z] + 1; kc++) {

	/* left hand edge */
	index = get_site_index(ic, 0, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < NVEL; p++) {
	    f_actual = distribution_f(index, p, nd);
	    f_expect = -1.0;

	    if (cv[p][Y] > 0) {
	      f_expect = offset[X];
	      if (cart_coords(dim) == 0) f_expect = L(X);
	    }
	  }

	  /* right hand edge */
	  jc = n_local[Y] + 1;
	  index = get_site_index(ic, jc, kc);

	  for (p = 0; p < NVEL; p++) {
	    f_actual = distribution_f(index, p, nd);
	    f_expect = -1.0;

	    if (cv[p][Y] < 0) {
	      f_expect = offset[X] + ic;
	      if (cart_coords(X) == cart_size(X) - 1) f_expect = 1.0;
	    }
	  }
	}

	/* Next site */
      }
    }

    /* Z-DIRECTION ? */

    /* Finished reduced check */
  }

  return;
}

/*****************************************************************************
 *
 *  on_edge
 *
 *  Returns 1 if on one of the twelve edges (including corners) of the domain,
 *          0  otherwise.
 *
 *****************************************************************************/

static int on_edge(int ic, int jc, int kc, int xmax, int ymax, int zmax) {

  int isedge = 0;

  if (ic == 0 && jc == 0) isedge = 1;
  if (ic == 0 && kc == 0) isedge = 1;
  if (jc == 0 && kc == 0) isedge = 1;

  if (ic == 0 && jc == ymax) isedge = 1;
  if (ic == 0 && kc == zmax) isedge = 1;

  if (jc == 0 && ic == xmax) isedge = 1;
  if (jc == 0 && kc == zmax) isedge = 1;

  if (kc == 0 && ic == xmax) isedge = 1;
  if (kc == 0 && jc == ymax) isedge = 1;

  if (ic == xmax && jc == ymax) isedge = 1;
  if (ic == xmax && kc == zmax) isedge = 1;
  if (jc == ymax && kc == zmax) isedge = 1;

  return isedge;
}
