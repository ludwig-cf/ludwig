/*****************************************************************************
 *
 *  test_lb_bc_outflow_rhou.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "cs_limits.h"
#include "lb_bc_outflow_rhou.h"

__host__ int test_lb_bc_outflow_rhou_create(pe_t * pe, cs_t * cs);
__host__ int test_lb_bc_outflow_rhou_update(pe_t * pe, cs_t * cs, int nvel);
__host__ int test_lb_bc_outflow_rhou_impose(pe_t * pe, cs_t * cs, int nvel);

/*****************************************************************************
 *
 *  test_lb_bc_outflow_rhou_suite
 *
 *****************************************************************************/

__host__ int test_lb_bc_outflow_rhou_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_lb_bc_outflow_rhou_create(pe, cs);
  test_lb_bc_outflow_rhou_update(pe, cs, NVEL);
  test_lb_bc_outflow_rhou_impose(pe, cs, NVEL);

  pe_info(pe, "PASS     ./unit/test_lb_bc_outflow_rhou\n");

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_bc_outflow_rhou_create
 *
 *****************************************************************************/

__host__ int test_lb_bc_outflow_rhou_create(pe_t * pe, cs_t * cs) {

  lb_bc_outflow_opts_t options = lb_bc_outflow_opts_default();
  lb_bc_outflow_rhou_t * outflow = NULL;
  
  assert(pe);
  assert(cs);

  lb_bc_outflow_rhou_create(pe, cs, &options, &outflow);

  assert(outflow);
  assert(outflow->pe == pe);
  assert(outflow->cs == cs);

  assert(outflow->super.func);
  assert(outflow->super.id == LB_BC_OUTFLOW_RHOU);

  assert(lb_bc_outflow_opts_valid(outflow->options));

  assert(outflow->nlink == 0);
  assert(outflow->linkp);
  assert(outflow->linki);
  assert(outflow->linkj);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_bc_outflow_rhou_update
 *
 *****************************************************************************/

__host__ int test_lb_bc_outflow_rhou_update(pe_t * pe, cs_t * cs, int nvel) {

  int nlocal[3] = {0};
  int ntotal[3] = {0};
  int noffset[3] = {0};

  lb_bc_outflow_opts_t options = {.nvel = nvel,
                                  .flow = {0, 1, 0},
				  .rho0 = 2.0};
  lb_bc_outflow_rhou_t * outflow = NULL;

  hydro_options_t hopts = hydro_options_default();
  hydro_t * hydro = NULL;

  assert(pe);
  assert(cs);

  lb_bc_outflow_rhou_create(pe, cs, &options, &outflow);
  hydro_create(pe, cs, NULL, &hopts, &hydro);

  /* The rho0 is fixed and comes from options.rho0, so just run update */

  lb_bc_outflow_rhou_update(outflow, hydro);

  cs_nlocal(cs, nlocal);
  cs_ntotal(cs, ntotal);
  cs_nlocal_offset(cs, noffset);

  if (noffset[Y] + nlocal[Y] == ntotal[Y]) {

    int jcbound = nlocal[Y] + 1;
    cs_limits_t limits = {1, nlocal[X], jcbound, jcbound, 1, nlocal[Z]};

    for (int ic = limits.imin; ic <= limits.imax; ic++) {
      for (int jc = limits.jmin; jc <= limits.jmax; jc++) {
	for (int kc = limits.kmin; kc <= limits.kmax; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  double rho = 0.0;

	  hydro_rho(hydro, index, &rho);
	  assert(fabs(rho - options.rho0) < DBL_EPSILON);
	}
      }
    }
  }

  hydro_free(hydro);
  lb_bc_outflow_rhou_free(outflow);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_bc_outflow_rhou_impose
 *
 *  This involves a halo swap for which the flow must be in the x-direction
 *  at the moment.
 *
 *****************************************************************************/

__host__ int test_lb_bc_outflow_rhou_impose(pe_t * pe, cs_t * cs, int nvel) {

  int ierr = 0;

  int nlocal[3] = {0};
  int ntotal[3] = {0};
  int noffset[3] = {0};

  lb_bc_outflow_opts_t options = {.nvel = nvel,
                                  .flow = {1, 0, 0},
                                  .rho0 = 3.0};
  lb_bc_outflow_rhou_t * outflow = NULL;
  lb_t * lb = NULL;

  hydro_options_t hopts = hydro_options_default();
  hydro_t * hydro = NULL;

  double u0[3] = {0.01, 0.0, 0.0}; /* Domain outflow in x-direction */

  assert(pe);
  assert(cs);

  lb_bc_outflow_rhou_create(pe, cs, &options, &outflow);
  hydro_create(pe, cs, NULL, &hopts, &hydro);

  {
    lb_data_options_t opts = lb_data_options_default();
    opts.ndim = NDIM;
    opts.nvel = nvel;
    lb_data_create(pe, cs, &opts, &lb);
  }


  /* Set some outflow velocities in the domain */

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  if (noffset[X] + nlocal[X] == ntotal[X]) {

    cs_limits_t limits = {nlocal[X], nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};

    for (int ic = limits.imin; ic <= limits.imax; ic++) {
      for (int jc = limits.jmin; jc <= limits.jmax; jc++) {
	for (int kc = limits.kmin; kc <= limits.kmax; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  hydro_u_set(hydro, index, u0);
	}
      }
    }
  }

  /* Run update */
  /* Check the resulting distributions at the outflow */

  lb_bc_outflow_rhou_update(outflow, hydro);
  lb_bc_outflow_rhou_impose(outflow, hydro, lb);

  if (noffset[X] + nlocal[X] == ntotal[X]) {

    int icbound = nlocal[X] + 1;
    cs_limits_t limits = {icbound, icbound, 1, nlocal[Y], 1, nlocal[Z]};

    for (int ic = limits.imin; ic <= limits.imax; ic++) {
      for (int jc = limits.jmin; jc <= limits.jmax; jc++) {
	for (int kc = limits.kmin; kc <= limits.kmax; kc++) {

	  int index = cs_index(cs, ic, jc, kc);

	  for (int p = 1; p < lb->model.nvel; p++) {

	    double f = 0.0;

	    if (lb->model.cv[p][X] != -1) continue;
	    if (jc + lb->model.cv[p][Y] < 1) continue;
	    if (kc + lb->model.cv[p][Z] < 1) continue;
	    if (jc + lb->model.cv[p][Y] > nlocal[Y]) continue;
	    if (kc + lb->model.cv[p][Z] > nlocal[Z]) continue;

	    lb_f(lb, index, p, LB_RHO, &f);

	    {
	      double ux   = u0[X];
	      double rho0 = outflow->options.rho0;
	      double fp   = lb->model.wv[p]*rho0*(1.0 - 3.0*ux + 3.0*ux*ux);
	      assert(fabs(f - fp) < DBL_EPSILON);
	      if (fabs(f - fp) > DBL_EPSILON) ierr += 1;
	    }
	  }
	}
      }
    }
  }

  lb_free(lb);
  hydro_free(hydro);
  lb_bc_outflow_rhou_free(outflow);

  return ierr;
}
