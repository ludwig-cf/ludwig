/*****************************************************************************
 *
 *  test_phi_bc_inflow_fixed.c
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "phi_bc_inflow_fixed.h"

__host__ int test_phi_bc_inflow_fixed_create(pe_t * pe, cs_t * cs);
__host__ int test_phi_bc_inflow_fixed_update(pe_t * pe, cs_t * cs);

/*****************************************************************************
 *
 *  test_phi_bc_inflow_fixed_suite
 *
 *****************************************************************************/

__host__ int test_phi_bc_inflow_fixed_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_phi_bc_inflow_fixed_create(pe, cs);
  test_phi_bc_inflow_fixed_update(pe, cs);

  pe_info(pe, "PASS     ./unit/test_phi_bc_inflow_fixed\n");

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_phi_bc_inflow_fixed_create
 *
 *****************************************************************************/

__host__ int test_phi_bc_inflow_fixed_create(pe_t * pe, cs_t * cs) {

  phi_bc_inflow_opts_t options = {.phi0 = 1.0, .flow = {1,0,0}};
  phi_bc_inflow_fixed_t * inflow = NULL;

  assert(pe);
  assert(cs);

  phi_bc_inflow_fixed_create(pe, cs, &options, &inflow);
  assert(inflow);

  assert(inflow->options.phi0    == options.phi0);    /* Really equal */
  assert(inflow->options.flow[0] == options.flow[0]);
  assert(inflow->options.flow[1] == options.flow[1]);
  assert(inflow->options.flow[2] == options.flow[2]);

  assert(inflow->super.func);
  assert(inflow->super.id == PHI_BC_INFLOW_FIXED);

  phi_bc_inflow_fixed_free(inflow);

  return 0;
}

/*****************************************************************************
 *
 *  test_phi_bc_inflow_fixed_update
 *
 *****************************************************************************/

__host__ int test_phi_bc_inflow_fixed_update(pe_t * pe, cs_t * cs) {

  int noffset[3] = {};
  phi_bc_inflow_opts_t options = {.phi0 = 999.999, .flow = {1,0,0}};
  phi_bc_inflow_fixed_t * inflow = NULL;

  field_t * phi = NULL;
  field_options_t opts = field_options_ndata_nhalo(1, 1);

  assert(pe);
  assert(cs);

  field_create(pe, cs, NULL, "phi", &opts, &phi);
  assert(phi);
  
  phi_bc_inflow_fixed_create(pe, cs, &options, &inflow);
  /* Run the update */
  {
    phi_bc_open_t * super = (phi_bc_open_t *) inflow;
    super->func->update(super, phi);
  }

  cs_nlocal_offset(cs, noffset);

  if (noffset[X] == 0) {
    /* Check x-inflow region */
    int nhalo = 0;
    int nlocal[3] = {};

    cs_nhalo(cs, &nhalo);
    cs_nlocal(cs, nlocal);

    for (int ic = 1 - nhalo; ic <= 0; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  double phibc = 0.0;
	  field_scalar(phi, index, &phibc);
	  assert(fabs(phibc - inflow->options.phi0) < DBL_EPSILON);
	}
      }
    }
  }

  {
    phi_bc_open_t * super = (phi_bc_open_t *) inflow;
    super->func->free(super);
  }
  field_free(phi);

  return 0;
}

