/*****************************************************************************
 *
 *  test_halo.c
 *
 *  This is a more rigourous test of the halo swap code for the
 *  distributions than appears in test model.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2007)
 *
 *  $Id: test_halo.c,v 1.4 2008-08-22 01:05:34 erlend Exp $
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h> /* for printf in test_halo_null */

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "tests.h"
#include "control.h"

static void test_halo_null(void);
static void test_halo(const int dim);
static void test_propagation();
int on_corner(int x, int y, int z, int mx, int my, int mz);

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

  RUN_read_input_file("input");
  init_control();
  coords_init();
  init_site();

  if(use_reduced_halos()) {
    info("Using reduced halos.\n");
    
    xfwd = calloc(NVEL, sizeof(int));
    xbwd = calloc(NVEL, sizeof(int));
    yfwd = calloc(NVEL, sizeof(int));
    ybwd = calloc(NVEL, sizeof(int));
    zfwd = calloc(NVEL, sizeof(int));
    zbwd = calloc(NVEL, sizeof(int));
    
    for(i=0; i<xcountcv; i++) {
      for(j=0; j<xdisp_fwd_cv[i]; j++) {
	for(k=0; k<xblocklens_cv[i]; k++) {
	  xfwd[xdisp_fwd_cv[i]+k] = 1;
	}
      }
    }
    
    for(i=0; i<xcountcv; i++) {
      for(j=0; j<xdisp_bwd_cv[i]; j++) {
	for(k=0; k<xblocklens_cv[i]; k++) {
	  xbwd[xdisp_bwd_cv[i]+k] = 1;
	}
      }
    }

    for(i=0; i<ycountcv; i++) {
      for(j=0; j<ydisp_fwd_cv[i]; j++) {
	for(k=0; k<yblocklens_cv[i]; k++) {
	  yfwd[ydisp_fwd_cv[i]+k] = 1;
	}
      }
    }

    for(i=0; i<ycountcv; i++) {
      for(j=0; j<ydisp_bwd_cv[i]; j++) {
	for(k=0; k<yblocklens_cv[i]; k++) {
	  ybwd[ydisp_bwd_cv[i]+k] = 1;
	}
      }
    }

    for(i=0; i<zcountcv; i++) {
      for(j=0; j<zdisp_fwd_cv[i]; j++) {
	for(k=0; k<zblocklens_cv[i]; k++) {
	  zfwd[zdisp_fwd_cv[i]+k] = 1;
	}
      }
    }

    for(i=0; i<zcountcv; i++) {
      for(j=0; j<zdisp_bwd_cv[i]; j++) {
	for(k=0; k<zblocklens_cv[i]; k++) {
	  zbwd[zdisp_bwd_cv[i]+k] = 1;
	}
      }
    }
  } else {
    info("Using full halos \n");
  }

  info("Test for null leakage...");
  test_halo_null();
  info("ok\n");

  info("Testing x-direction swap...");
  test_halo(X);
  info("ok\n");

  info("Testing y-direction swap...");
  test_halo(Y);
  info("ok\n");

  info("Testing z-direction swap...");
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

  get_N_local(n_local);

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);

  /* Set entire distribution (all sites including halos) */

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

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

	index = index_site(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, 0.0);
	  set_g_at_site(index, p, 0.0);
	}

      }
    }
  }

  /* Swap */

  halo_site();

  /* Check everywhere */

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  f_actual = get_f_at_site(index, p);
	  g_actual = get_g_at_site(index, p);
	  if( n[X] >= 1 && n[Y] >= 1 && n[Z] >= 1			\
	      && n[X] <= n_local[X] && n[Y] <= n_local[Y] && n[Z] <= n_local[Z]) {
	    /* everything should still be zero inside the lattice */
	    test_assert(fabs(f_actual - 0.0) < TEST_DOUBLE_TOLERANCE);
	  } else {
	    if(!use_reduced_halos()) {
	      /* everything should be zero on the halo */
	      test_assert(fabs(f_actual - 0.0) < TEST_DOUBLE_TOLERANCE);
	    } else {

	      /* reduced: some velocities will be non-zero */
	      if( n[X] == 0 &&
		  !on_corner(n[X], n[Y], n[Z],n_local[X]+1, n_local[Y]+1, n_local[Z]+1)
		  ) { /* fwd */
		if(fabs(f_actual-0.0) > TEST_DOUBLE_TOLERANCE) {
		  /* it's a one, should it be? 
		   NB: this means nothing got swapped for this vel */
		  if(xfwd[p] > 0) { /* check that */
		    printf("Error n[X]=%d, n[Y]=%d, n[Z]=%d, p=%d\n",\
			   n[X], n[Y], n[Z], p);
		  }

		}
	      }

	      if( n[X] > n_local[X] &&
		  !on_corner(n[X], n[Y], n[Z],n_local[X]+1, n_local[Y]+1, n_local[Z]+1)
		  ) { /* bwd*/
		if(fabs(f_actual-0.0) > TEST_DOUBLE_TOLERANCE) {
		  /* it's a one, should it be? 
		   NB: this means nothing got swapped for this vel */
		  if(xbwd[p] > 0) { /* check that */
		    printf("Error n[X]=%d, n[Y]=%d, n[Z]=%d, p=%d\n",\
			   n[X], n[Y], n[Z], p);
		  }

		}
	      }
	      
	      if( n[Y] == 0 &&						\
		  !on_corner(n[X], n[Y], n[Z],n_local[X]+1, n_local[Y]+1, n_local[Z]+1)
		  ) { /* fwd */
		if(fabs(f_actual-0.0) > TEST_DOUBLE_TOLERANCE) {
		  /* it's a one, should it be? 
		     NB: this means nothing got swapped for this vel */
		  if(yfwd[p] > 0) { /* check that */
		    printf("Error n[X]=%d, n[Y]=%d, n[Z]=%d, p=%d\n",	\
			   n[X], n[Y], n[Z], p);
		  }
		  
		}
	      }

	      if( n[Y] > n_local[Y] && 
		  !on_corner(n[X], n[Y], n[Z],n_local[X]+1, n_local[Y]+1, n_local[Z]+1)
		  ) { /* fwd */
		if(fabs(f_actual-0.0) > TEST_DOUBLE_TOLERANCE) {
		  /* it's a one, should it be? 
		   NB: this means nothing got swapped for this vel */
		  if(ybwd[p] > 0) { /* check that */
		    printf("Error n[X]=%d, n[Y]=%d, n[Z]=%d, p=%d\n",\
			   n[X], n[Y], n[Z], p);
		  }

		}
	      }
	      
	      if( n[Z] == 0 && \
		  !on_corner(n[X], n[Y], n[Z],n_local[X]+1, n_local[Y]+1, n_local[Z]+1)
		  ) { /* fwd */
		if(fabs(f_actual-0.0) > TEST_DOUBLE_TOLERANCE) {
		  /* it's a one, should it be? 
		   NB: this means nothing got swapped for this vel */
		  if(zfwd[p] > 0) { /* check that */
		    printf("Error n[X]=%d, n[Y]=%d, n[Z]=%d, p=%d\n",\
			   n[X], n[Y], n[Z], p);
		  }

		}
	      }

	      if( n[Z] > n_local[Z] &&				\
		  !on_corner(n[X], n[Y], n[Z],n_local[X]+1, n_local[Y]+1, n_local[Z]+1)
		  ) { /* fwd */
		if(fabs(f_actual-0.0) > TEST_DOUBLE_TOLERANCE) {
		  /* it's a one, should it be? 
		   NB: this means nothing got swapped for this vel */
		  if(zbwd[p] > 0) { /* check that */
		    printf("Error n[X]=%d, n[Y]=%d, n[Z]=%d, p=%d\n",\
			   n[X], n[Y], n[Z], p);
		  }

		}
	      }

	    }
	    
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
 *****************************************************************************/

void test_halo(int dim) {

  int n_local[3], n[3];
  int offset[3];
  int index, p, d;

  double f_expect, f_actual;

  test_assert(dim == X || dim == Y || dim == Z);
  int* fwd;
  int* bwd;
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

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, 0.0);
	}

      }
    }
  }

  /* Check neighbours in the given direction */

  for (n[X] = 1; n[X] <= n_local[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= n_local[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= n_local[Z]; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	if (n[X] == 1 || n[X] == n_local[X] ||
	    n[Y] == 1 || n[Y] == n_local[Y] ||
	    n[Z] == 1 || n[Z] == n_local[Z]) {

	  for (p = 0; p < NVEL; p++) {
	    set_f_at_site(index, p, offset[dim] + n[dim]);
	  }
	}

      }
    }
  }

  halo_site();

  /* Check the results (all sites).
   * The halo regions should contain a copy of the above, while the
   * interior sites are unchanged */

  for (n[X] = 0; n[X] <= n_local[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= n_local[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= n_local[Z] + 1; n[Z]++) {

	index = index_site(n[X], n[Y], n[Z]);

	for (d = 0; d < 3; d++) {

	  /* 'Left' side */
	  if (dim == d && n[d] == 0) {

	    f_expect = offset[dim];
	    if (cart_coords(dim) == 0) f_expect = L(dim);

	    for (p = 0; p < NVEL; p++) {
	      if(!use_reduced_halos()) {

		f_actual = get_f_at_site(index, p);
		test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);

	      } else {

		if(fwd[p] > 0 && \
		   !on_corner(n[X], n[Y], n[Z], n_local[X]+1, n_local[Y]+1, n_local[Z]+1)) 
		  {
		    f_actual = get_f_at_site(index, p);
		    if(fabs(f_actual-f_expect) > TEST_DOUBLE_TOLERANCE) {
		      printf("ERROR: dim = %d, fwd[%d]=%d, x=%d,y=%d,z=%d\n", \
			     dim, p, fwd[p], n[X], n[Y], n[Z]);
		    }
		    test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
		}

	      }

	    }
	  }

	  /* 'Right' side */
	  if (dim == d && n[d] == n_local[d] + 1) {

	    f_expect = offset[dim] + n_local[dim] + 1.0;
	    if (cart_coords(dim) == cart_size(dim) - 1) f_expect = 1.0;

	    for (p = 0; p < NVEL; p++) {
	      if(!use_reduced_halos()) {
		f_actual = get_f_at_site(index, p);
		test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
	      } else {
		if(bwd[p] > 0 && \
		   !on_corner(n[X], n[Y], n[Z], n_local[X]+1, n_local[Y]+1, n_local[Z]+1))
		  {
		    f_actual = get_f_at_site(index, p);
		    if(fabs(f_actual-f_expect) > TEST_DOUBLE_TOLERANCE) {
		      printf("ERROR: dim = %d, bwd[%d]=%d, x=%d,y=%d,z=%d\n", \
			     dim, p, bwd[p], n[X], n[Y], n[Z]);
		    }
		    test_assert(fabs(f_actual - f_expect) < TEST_DOUBLE_TOLERANCE);
		  }
	      }

	    }
	  }

	}

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
int on_corner(int x, int y, int z, \
	      int mx, int my, int mz) {
  /* on the axes */
  if( abs(x) + abs(y) == 0 || \
      abs(x) + abs(z) == 0 || \
      abs(y) + abs(z) == 0 )
    {
      return 1;
    }

  /* opposite corners from axes */
  if( x == mx && y == my ||
      x == mx && z == mz ||
      y == my && z == mz)
    {
      return 1;
    }
  
  if( x == 0 && y == my ||
      x == 0 && z == mz ||
      y == 0 && x == mx ||
      y == 0 && z == mz ||
      z == 0 && x == mx ||
      z == 0 && y == my)
    {
      return 1;
    }

  return 0;
}
