/*****************************************************************************
 *
 *  test_prop
 *
 *  Test propagation stage.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2024 Ths University of Edinburgh
 *
 *  Contributing authors: 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "memory.h"
#include "propagation.h"
#include "tests.h"

__host__ int do_test_velocity(pe_t * pe, cs_t * cs, int ndist,
			      lb_halo_enum_t halo);
__host__ int do_test_source_destination(pe_t * pe, cs_t * cs, int ndist,
					lb_halo_enum_t halo);

/*****************************************************************************
 *
 *  test_lb_prop_suite
 *
 *****************************************************************************/

int test_lb_prop_suite(void) {

  int ndevice = 0;
  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  do_test_velocity(pe, cs, 1, LB_HALO_TARGET);
  do_test_velocity(pe, cs, 2, LB_HALO_TARGET);
  if (ndevice == 0) {
    do_test_velocity(pe, cs, 1, LB_HALO_OPENMP_FULL);
    do_test_velocity(pe, cs, 1, LB_HALO_OPENMP_REDUCED);
  }

  do_test_source_destination(pe, cs, 1, LB_HALO_TARGET);
  do_test_source_destination(pe, cs, 2, LB_HALO_TARGET);
  if (ndevice == 0) {
    do_test_source_destination(pe, cs, 1, LB_HALO_OPENMP_FULL);
    do_test_source_destination(pe, cs, 1, LB_HALO_OPENMP_REDUCED);
  }

  pe_info(pe, "PASS     ./unit/test_prop\n");
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_velocity
 *
 *  Check each distribution ends up with the same velocity index.
 *  This relies on the halo exchange working properly.
 *
 *****************************************************************************/

int do_test_velocity(pe_t * pe, cs_t * cs, int ndist, lb_halo_enum_t halo) {

  int nlocal[3];
  int ic, jc, kc, index, p;
  int nd;
  double f_actual;

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);
  assert(ndist == 1 || ndist == 2);

  options.ndim = NDIM;
  options.nvel = NVEL;
  options.ndist = ndist;
  options.halo  = halo;

  lb_data_create(pe, cs, &options, &lb);
  assert(lb);

  cs_nlocal(cs, nlocal);

  /* Set test values */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f_set(lb, index, p, nd, 1.0*(p + nd*lb->model.nvel));
	  }
	}

      }
    }
  }

  /* Halo swap, and make sure values are on device */

  lb_memcpy(lb, tdpMemcpyHostToDevice);
  lb_halo(lb);

  lb_propagation(lb);
  lb_memcpy(lb, tdpMemcpyDeviceToHost);

  /* Test */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f(lb, index, p, nd, &f_actual);
	    assert(fabs(f_actual - 1.0*(p + nd*lb->model.nvel)) < DBL_EPSILON);
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
 *  do_test_source_destination
 *
 *  Check each element of the distribution has propagated exactly one
 *  lattice spacing in the appropriate direction.
 *
 *  We use the global index as the test of the soruce.
 *  
 *****************************************************************************/

int do_test_source_destination(pe_t * pe, cs_t * cs, int ndist,
			       lb_halo_enum_t halo) {

  int nlocal[3], offset[3];
  int ntotal[3];
  int ic, jc, kc, index, p;
  int nd;
  int isource, jsource, ksource;
  double f_actual, f_expect;
  double ltot[3];

  lb_data_options_t options = lb_data_options_default();
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);
  assert(ndist == 1 || ndist == 2);

  options.ndim = NDIM;
  options.nvel = NVEL;
  options.halo = halo;
  options.ndist = ndist;

  lb_data_create(pe, cs, &options, &lb);
  assert(lb);

  cs_ltot(cs, ltot);
  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, offset);

  /* Set test values */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);

	f_actual = ltot[Y]*ltot[Z]*(offset[X] + ic) +
	  ltot[Z]*(offset[Y] + jc) + (offset[Z] + kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f_set(lb, index, p, nd, f_actual);
	  }
	}

      }
    }
  }

  /* Initial values update to device */
  lb_memcpy(lb, tdpMemcpyHostToDevice);
  lb_halo(lb);

  lb_propagation(lb);
  lb_memcpy(lb, tdpMemcpyDeviceToHost);

  /* Test */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);

	for (nd = 0; nd < ndist; nd++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    isource = offset[X] + ic - lb->model.cv[p][X];
	    if (isource == 0) isource += ntotal[X];
	    if (isource == ntotal[X] + 1) isource = 1;
	    jsource = offset[Y] + jc - lb->model.cv[p][Y];
	    if (jsource == 0) jsource += ntotal[Y];
	    if (jsource == ntotal[Y] + 1) jsource = 1;
	    ksource = offset[Z] + kc - lb->model.cv[p][Z];
	    if (ksource == 0) ksource += ntotal[Z];
	    if (ksource == ntotal[Z] + 1) ksource = 1;

	    f_expect = ltot[Y]*ltot[Z]*isource + ltot[Z]*jsource + ksource;
	    lb_f(lb, index, p, nd, &f_actual);

	    /* In case of d2q9, propagation is only for kc = 1 */
	    if (lb->model.ndim == 2 && kc > 1) f_actual = f_expect;

	    assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	  }
	}

	/* Next site */
      }
    }
  }

  lb_free(lb);

  return 0;
}
