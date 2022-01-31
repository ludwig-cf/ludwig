/*****************************************************************************
 *
 *  test_phi_bc_outflow_free.c
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "phi_bc_outflow_free.h"

__host__ int test_phi_bc_outflow_free_create(pe_t * pe, cs_t * cs);
__host__ int test_phi_bc_outflow_free_update(pe_t * pe, cs_t * cs);

/*****************************************************************************
 *
 *  test_phi_bc_outflow_free_suite
 *
 *****************************************************************************/

__host__ int test_phi_bc_outflow_free_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_phi_bc_outflow_free_create(pe, cs);
  test_phi_bc_outflow_free_update(pe, cs);

  pe_info(pe, "PASS     ./unit/phi_bc_outflow_free\n");
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_phi_bc_outflow_free_create
 *
 *****************************************************************************/

__host__ int test_phi_bc_outflow_free_create(pe_t * pe, cs_t * cs) {

  phi_bc_outflow_opts_t options = {.flow = {1,0,0}};
  phi_bc_outflow_free_t * outflow = NULL;

  assert(pe);
  assert(cs);

  phi_bc_outflow_free_create(pe, cs, &options, &outflow);
  assert(outflow);

  assert(outflow->options.flow[X] == options.flow[X]);
  assert(outflow->options.flow[Y] == options.flow[Y]);
  assert(outflow->options.flow[Z] == options.flow[Z]);

  assert(outflow->super.func);
  assert(outflow->super.id == PHI_BC_OUTFLOW_FREE);

  phi_bc_outflow_free_free(outflow);

  return 0;
}

/*****************************************************************************
 *
 *  test_phi_bc_outflow_free_update
 *
 *****************************************************************************/

__host__ int test_phi_bc_outflow_free_update(pe_t * pe, cs_t * cs) {

  int nhalo = -1;
  int nlocal[3] = {};
  int ntotal[3] = {};
  int noffset[3] = {};
  
  phi_bc_outflow_opts_t options = {.flow = {1,0,0}};
  phi_bc_outflow_free_t * outflow = NULL;

  field_t * phi = NULL;
  field_options_t opts = field_options_ndata_nhalo(1, 1);
  
  assert(pe);
  assert(cs);

  field_create(pe, cs, NULL, "phi", &opts, &phi);

  /* Provide some initial conditions in the domain proper,
   * with some artificial data. */

  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);
  cs_nhalo(cs, &nhalo);

  {
    int ic = nlocal[X];
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(cs, ic, jc, kc);
	field_scalar_set(phi, index, 1.0*index);
      }
    }
  }

  /* Run the update */
  phi_bc_outflow_free_create(pe, cs, &options, &outflow);
  {
    phi_bc_open_t * bc = (phi_bc_open_t *) outflow;
    bc->func->update(bc, phi);
  }

  /* Check */
  if (noffset[X] + nlocal[X] == ntotal[X]) {
    for (int ic = nlocal[X] + 1; ic <= nlocal[X] + nhalo; ic++) {
      for (int jc = 1; jc <= nlocal[Y]; jc++) {
	for (int kc = 1; kc <= nlocal[Z]; kc++) {
	  int index0 = cs_index(cs, nlocal[X], jc, kc);
	  int index1 = cs_index(cs, ic, jc, kc);
	  double phi0 = 0.0;
	  double phi1 = 0.0;
	  field_scalar(phi, index0, &phi0);
	  field_scalar(phi, index1, &phi1);
	  assert(fabs(phi1 - phi0) < DBL_EPSILON);
	}
      }
    }
  }

  {
    phi_bc_open_t * bc = (phi_bc_open_t *) outflow;
    bc->func->free(bc);
  }
  field_free(phi);

  return 0;
}
