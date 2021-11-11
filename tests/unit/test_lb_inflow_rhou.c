/*****************************************************************************
 *
 *  test_lb_inflow_rhou.c
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "lb_inflow_rhou.h"

/* Tests */

__host__ int test_lb_inflow_rhou_create(pe_t * pe, cs_t * cs);
__host__ int test_lb_inflow_rhou_update(pe_t * pe, cs_t * cs);
__host__ int test_lb_inflow_rhou_impose(pe_t * pe, cs_t * cs);

/*****************************************************************************
 *
 *  test_lb_inflow_rhou_suite
 *
 *****************************************************************************/

__host__ int test_lb_inflow_rhou_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_lb_inflow_rhou_create(pe, cs);
  test_lb_inflow_rhou_update(pe, cs);
  test_lb_inflow_rhou_impose(pe, cs);

  pe_info(pe, "PASS     ./unit/test_lb_inflow_rhou\n");

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_inflow_rhou_create
 *
 *****************************************************************************/

__host__ int test_lb_inflow_rhou_create(pe_t * pe, cs_t * cs) {

  lb_openbc_options_t options = lb_openbc_options_default();
  lb_inflow_rhou_t * inflow = NULL;

  assert(pe);
  assert(cs);

  lb_inflow_rhou_create(pe, cs, &options, &inflow);

  assert(inflow);
  assert(inflow->pe == pe);
  assert(inflow->cs = cs);

  assert(inflow->super.func);
  assert(inflow->super.id == LB_OPEN_BC_INFLOW_RHOU);

  /* Check options */

  assert(inflow->options.inflow);

  /* Default options given no links */

  assert(inflow->nlink == 0);
  assert(inflow->linkp);
  assert(inflow->linki);
  assert(inflow->linkj);

  lb_inflow_rhou_free(inflow);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_inflow_rhou_update
 *
 *****************************************************************************/

typedef struct limits_s {
  int imin, imax;
  int jmin, jmax;
  int kmin, kmax;
} limits_t;

/*****************************************************************************
 *
 *  test_lb_inflow_rhou_update
 *
 *****************************************************************************/

__host__ int test_lb_inflow_rhou_update(pe_t * pe, cs_t * cs) {

  int nlocal[3] = {};
  int noffset[3] = {};
  double rho0 = 2.0;

  lb_openbc_options_t options = {.flow = {1, 0, 0},
                                 .u0   = {-1.0,-2.0,-3.0}};
  lb_inflow_rhou_t * inflow = NULL;
  hydro_t * hydro = NULL;

  assert(pe);
  assert(cs);

  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  lb_inflow_rhou_create(pe, cs, &options, &inflow);
  hydro_create(pe, cs, NULL, 1, &hydro);

  /* Set the relevant domain values (rho only here) */

  if (noffset[X] == 0) {

    limits_t limits = {1, 1, 1, nlocal[Y], 1, nlocal[Z]};

    for (int ic = limits.imin; ic <= limits.imax; ic++) {
      for (int jc = limits.jmin; jc <= limits.jmax; jc++) {
	for (int kc = limits.kmin; kc <= limits.kmax; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  hydro_rho_set(hydro, index, rho0);
	}
      }
    }
  }

  /* Run the update. */
  /* Check rho, u = 0 in the boundary region are set */

  lb_inflow_rhou_update(inflow, hydro);

  if (noffset[X] == 0) {

    limits_t limits = {0, 0, 1, nlocal[Y], 1, nlocal[Z]};

    for (int ic = limits.imin; ic <= limits.imax; ic++) {
      for (int jc = limits.jmin; jc <= limits.jmax; jc++) {
	for (int kc = limits.kmin; kc <= limits.kmax; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  double rho = 0.0;
	  double u[3] = {};

	  hydro_rho(hydro, index, &rho);
	  hydro_u(hydro, index, u);

	  assert(fabs(rho - rho0) < DBL_EPSILON);
	  assert(fabs(u[X] - inflow->options.u0[X]) < DBL_EPSILON);
	  assert(fabs(u[Y] - inflow->options.u0[Y]) < DBL_EPSILON);
	  assert(fabs(u[Z] - inflow->options.u0[Z]) < DBL_EPSILON);
	}
      }
    }
  }

  hydro_free(hydro);
  lb_inflow_rhou_free(inflow);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_inflow_rhou_impose
 *
 *****************************************************************************/

__host__ int test_lb_inflow_rhou_impose(pe_t * pe, cs_t * cs) {

  int ierr = 0;

  int nlocal[3] = {};
  int ntotal[3] = {};
  int noffset[3] = {};

  double rho0 = 1.0;

  lb_openbc_options_t options = {.nvel = NVEL,
                                 .flow = {1, 0, 0},
                                 .u0   = {0.01, 0.0, 0.0}};
  lb_inflow_rhou_t * inflow = NULL;
  hydro_t * hydro = NULL;
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  /* Set up. */

  cs_nlocal(cs, nlocal);
  cs_ntotal(cs, ntotal);
  cs_nlocal_offset(cs, noffset);

  lb_inflow_rhou_create(pe, cs, &options, &inflow);
  hydro_create(pe, cs, NULL, 1, &hydro);
  lb_create(pe, cs, &lb);
  lb_init(lb);

  /* Set the relevant domain values (rho only here) */

  if (noffset[X] == 0) {

    limits_t limits = {1, 1, 1, nlocal[Y], 1, nlocal[Z]};

    for (int ic = limits.imin; ic <= limits.imax; ic++) {
      for (int jc = limits.jmin; jc <= limits.jmax; jc++) {
	for (int kc = limits.kmin; kc <= limits.kmax; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  hydro_rho_set(hydro, index, rho0);
	}
      }
    }
  }

  lb_inflow_rhou_update(inflow, hydro);
  lb_inflow_rhou_impose(inflow, hydro, lb);

  /* Check relevant f_i are set correctly. */
  /* So, for each site in the boundary region, check ingoing distributions. */

  if (noffset[X] == 0) {
    limits_t limits = {0, 0, 1, nlocal[Y], 1, nlocal[Z]};

    for (int ic = limits.imin; ic <= limits.imax; ic++) {
      for (int jc = limits.jmin; jc <= limits.jmax; jc++) {
	for (int kc = limits.kmin; kc <= limits.kmax; kc++) {
	  for (int p = 1; p < lb->model.nvel; p++) {

	    int index = cs_index(cs, ic, jc, kc);
	    double f = 0.0;

	    /* Only distributions ending in the local domain ... */
	    if (lb->model.cv[p][X] != +1) continue;
	    if (noffset[Y] + jc + lb->model.cv[p][Y] < 1) continue;
	    if (noffset[Z] + kc + lb->model.cv[p][Z] < 1) continue;
	    if (noffset[Y] + jc + lb->model.cv[p][Y] > ntotal[Y]) continue;
	    if (noffset[Z] + kc + lb->model.cv[p][Z] > ntotal[Z]) continue;

	    lb_f(lb, index, p, LB_RHO, &f);
	    {
	      double ux = inflow->options.u0[X];
	      double fp = lb->model.wv[p]*rho0*(1.0 + 3.0*ux + 3.0*ux*ux);
	      assert(fabs(f - fp) < DBL_EPSILON);
	      ierr += (fabs(f - fp) > DBL_EPSILON);
	    }
	  }
	}
      }
    }
  }

  lb_free(lb);
  hydro_free(hydro);
  lb_inflow_rhou_free(inflow);

  return ierr;
}
