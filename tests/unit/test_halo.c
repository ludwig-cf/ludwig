/*****************************************************************************
 *
 *  test_halo.c
 *
 *  This is a more rigourous test of the halo swap code for the
 *  distributions than appears in test model.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "lb_data.h"
#include "control.h"
#include "tests.h"

int test_lb_halo1(pe_t * pe, cs_t * cs, int ndim, int nvel);
int do_test_halo_null(pe_t * pe, cs_t * cs, const lb_data_options_t * opts);
int do_test_halo(pe_t * pe, cs_t * cs, int dim, const lb_data_options_t * opts);

/*****************************************************************************
 *
 *  test_halo_suite
 *
 *****************************************************************************/

int test_halo_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  /* Use a 2d system for ndim = 2, nvel = 9 */
  test_lb_halo1(pe, cs, 3, 15);
  pe_info(pe, "PASS     ./unit/test_halo 15\n");
  test_lb_halo1(pe, cs, 3, 19);
  pe_info(pe, "PASS     ./unit/test_halo 19\n");
  test_lb_halo1(pe, cs, 3, 27);

  pe_info(pe, "PASS     ./unit/test_halo\n");
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_halo
 *
 *****************************************************************************/

int test_lb_halo1(pe_t * pe, cs_t * cs, int ndim, int nvel) {

  lb_data_options_t opts = lb_data_options_default();

  opts.ndim  = ndim;
  opts.nvel  = nvel;
  opts.ndist = 1;
  opts.halo  = LB_HALO_TARGET;

  do_test_halo_null(pe, cs, &opts);
  do_test_halo(pe, cs, X, &opts);
  do_test_halo(pe, cs, Y, &opts);
  do_test_halo(pe, cs, Z, &opts);

  opts.ndist = 1;
  opts.halo  = LB_HALO_OPENMP_FULL;

  do_test_halo_null(pe, cs, &opts);
  do_test_halo(pe, cs, X, &opts);
  do_test_halo(pe, cs, Y, &opts);
  do_test_halo(pe, cs, Z, &opts);

  opts.ndist = 1;
  opts.halo  = LB_HALO_OPENMP_REDUCED;

  do_test_halo_null(pe, cs, &opts);

  opts.ndist = 2;
  opts.halo = LB_HALO_TARGET;

  do_test_halo_null(pe, cs, &opts);
  do_test_halo(pe, cs, X, &opts);
  do_test_halo(pe, cs, Y, &opts);
  do_test_halo(pe, cs, Z, &opts);

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

int do_test_halo_null(pe_t * pe, cs_t * cs, const lb_data_options_t * opts) {

  int nlocal[3], n[3];
  int index, nd, p;
  int nhalo;
  int nextra;
  double f_actual;

  lb_t * lb = NULL;

  assert(pe);
  assert(cs);
  assert(opts);

  cs_nhalo(cs, &nhalo);
  nextra = nhalo - 1;

  lb_data_create(pe, cs, opts, &lb);

  cs_nlocal(cs, nlocal);

  /* Set entire distribution (all sites including halos) to 1.0 */

  for (n[X] = 1 - nextra; n[X] <= nlocal[X] + nextra; n[X]++) {
    for (n[Y] = 1 - nextra; n[Y] <= nlocal[Y] + nextra; n[Y]++) {
      for (n[Z] = 1 - nextra; n[Z] <= nlocal[Z] + nextra; n[Z]++) {

	index = cs_index(cs, n[X], n[Y], n[Z]);

	for (nd = 0; nd < lb->ndist; nd++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f_set(lb, index, p, nd, 1.0);
	  }
	}

      }
    }
  }

  /* Zero interior */

  for (n[X] = 1; n[X] <= nlocal[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= nlocal[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= nlocal[Z]; n[Z]++) {

	index = cs_index(cs, n[X], n[Y], n[Z]);

	for (nd = 0; nd < lb->ndist; nd++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f_set(lb, index, p, nd, 0.0);
	  }
	}

      }
    }
  }

  /* Swap */

  lb_halo(lb);

  /* Check everywhere in the interior still zero */

  for (n[X] = 1; n[X] <= nlocal[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= nlocal[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= nlocal[Z]; n[Z]++) {

	index = cs_index(cs, n[X], n[Y], n[Z]);

	for (nd = 0; nd < lb->ndist; nd++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f(lb, index, p, nd, &f_actual);

	    /* everything should still be zero inside the lattice */
	    test_assert(fabs(f_actual - 0.0) < DBL_EPSILON);
	  }
	}

      }
    }
  }

  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_halo
 *
 *  Test the halo swap for the distributions for coordinate direction dim.
 *
 *  Note that the reduced halo swaps are only meaningful in
 *  parallel. They will automatically work in serial.
 *
 *****************************************************************************/

int do_test_halo(pe_t * pe, cs_t * cs, int dim, const lb_data_options_t * opts) {

  int ndevice = 0;
  int nhalo;
  int nlocal[3], n[3];
  int offset[3];
  int mpi_cartsz[3];
  int mpi_cartcoords[3];
  int nd;
  int nextra;
  int index, p, d;

  double ltot[3];
  double f_expect, f_actual;
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);
  assert(dim == X || dim == Y || dim == Z);
  assert(opts);

  tdpGetDeviceCount(&ndevice);

  lb_data_create(pe, cs, opts, &lb);

  cs_nhalo(cs, &nhalo);
  nextra = nhalo;

  cs_ltot(cs, ltot);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, offset);
  cs_cartsz(cs, mpi_cartsz);
  cs_cart_coords(cs, mpi_cartcoords);

  /* Zero entire distribution (all sites including halos) */

  for (n[X] = 1 - nextra; n[X] <= nlocal[X] + nextra; n[X]++) {
    for (n[Y] = 1 - nextra; n[Y] <= nlocal[Y] + nextra; n[Y]++) {
      for (n[Z] = 1 - nextra; n[Z] <= nlocal[Z] + nextra; n[Z]++) {

	index = cs_index(cs, n[X], n[Y], n[Z]);

	for (nd = 0; nd < lb->ndist; nd++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f_set(lb, index, p, nd, -1.0);
	  }
	}

      }
    }
  }

  /* Set the interior sites to get swapped with a value related to
   * absolute position */

  for (n[X] = 1; n[X] <= nlocal[X]; n[X]++) {
    for (n[Y] = 1; n[Y] <= nlocal[Y]; n[Y]++) {
      for (n[Z] = 1; n[Z] <= nlocal[Z]; n[Z]++) {

	index = cs_index(cs, n[X], n[Y], n[Z]);

	if (n[X] <= nhalo || n[X] > nlocal[X] - nhalo ||
	    n[Y] <= nhalo || n[Y] > nlocal[Y] - nhalo ||
	    n[Z] <= nhalo || n[Z] > nlocal[Z] - nhalo) {

	  for (nd = 0; nd < lb->ndist; nd++) {
	    for (p = 0; p < lb->model.nvel; p++) {
	      lb_f_set(lb, index, p, nd, 1.0*(offset[dim] + n[dim]));
	    }
	  }
	}

      }
    }
  }

  lb_memcpy(lb, tdpMemcpyHostToDevice);
  lb_halo(lb);

  /* Don't overwrite the host version if not device swap */
  if (ndevice && lb->opts.halo == LB_HALO_TARGET) {
    lb_memcpy(lb, tdpMemcpyDeviceToHost);
  }

  /* Check the results (all sites for distribution halo).
   * The halo regions should contain a copy of the above, while the
   * interior sites are unchanged */

  /* Note the distribution halo swaps are always width 1, irrespective
   * of nhalo */

  for (n[X] = 0; n[X] <= nlocal[X] + 1; n[X]++) {
    for (n[Y] = 0; n[Y] <= nlocal[Y] + 1; n[Y]++) {
      for (n[Z] = 0; n[Z] <= nlocal[Z] + 1; n[Z]++) {

	index = cs_index(cs, n[X], n[Y], n[Z]);

	for (nd = 0; nd < lb->ndist; nd++) {
	  for (d = 0; d < 3; d++) {

	    /* 'Left' side */
	    if (dim == d && n[d] == 0) {

	      f_expect = offset[dim];
	      if (mpi_cartcoords[dim] == 0) f_expect = ltot[dim];

	      for (p = 0; p < lb->model.nvel; p++) {
		lb_f(lb, index, p, nd, &f_actual);
		test_assert(fabs(f_actual-f_expect) < DBL_EPSILON);
	      }
	    }

	    /* 'Right' side */
	    if (dim == d && n[d] == nlocal[d] + 1) {

	      f_expect = offset[dim] + nlocal[dim] + 1.0;
	      if (mpi_cartcoords[dim] == mpi_cartsz[dim] - 1) f_expect = 1.0;

	      for (p = 0; p < lb->model.nvel; p++) {
		lb_f(lb, index, p, nd, &f_actual);
		test_assert(fabs(f_actual-f_expect) < DBL_EPSILON);
	      }
	    }
	  }
	}
	/* Next site */
      }
    }
  }

  lb_free(lb);

  return 0;
}
