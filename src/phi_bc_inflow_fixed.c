/*****************************************************************************
 *
 *  phi_bc_inflow_fixed.c
 *
 *  A fixed inflow boundary condition which sets a uniformly constant
 *  phi_0 at the boundary region (to the current depth of the halo).
 *
 *  This is intended to propagate into the compution of the gradient
 *  in the boundary region (grad phi ~ 0 and \nabla^2 phi ~ 0) which
 *  should represent a constant chemical potential, and a constant
 *  stress on the fluid.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and 
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "phi_bc_inflow_fixed.h"

static const phi_bc_open_vtable_t vt_ = {
  (phi_bc_open_free_ft)   phi_bc_inflow_fixed_free,
  (phi_bc_open_update_ft) phi_bc_inflow_fixed_update
};

/*****************************************************************************
 *
 *  phi_bc_inflow_fixed_create
 *
 *****************************************************************************/

__host__ int phi_bc_inflow_fixed_create(pe_t * pe, cs_t * cs,
					const phi_bc_inflow_opts_t * options,
					phi_bc_inflow_fixed_t ** inflow) {

  phi_bc_inflow_fixed_t * bc = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(inflow);

  bc = (phi_bc_inflow_fixed_t *) calloc(1, sizeof(phi_bc_inflow_fixed_t));
  assert(bc);
  if (bc == NULL) pe_fatal(pe, "Failed to allocate phi_bc_inflow_fixed_t\n");

  /* Pointers and superclass */

  bc->pe = pe;
  bc->cs = cs;

  bc->super.func = &vt_;
  bc->super.id   = PHI_BC_INFLOW_FIXED;

  bc->options = *options;

  *inflow = bc;

  return 0;
}

/*****************************************************************************
 *
 *  phi_bc_inflow_fixed_free
 *
 *****************************************************************************/

__host__ int phi_bc_inflow_fixed_free(phi_bc_inflow_fixed_t * inflow) {

  assert(inflow);

  free(inflow);

  return 0;
}

/*****************************************************************************
 *
 *  phi_bc_inflow_fixed_update
 *
 *****************************************************************************/

__host__ int phi_bc_inflow_fixed_update(phi_bc_inflow_fixed_t * inflow,
					field_t * phi) {
  int id = -1;
  int nhalo = 0;
  int nlocal[3] = {};
  int noffset[3] = {};
  cs_t * cs = NULL;

  assert(inflow);
  assert(phi);

  /* Set in inflow region being X, Y, or Z. */

  if (inflow->options.flow[X]) id = X;
  if (inflow->options.flow[Y]) id = Y;
  if (inflow->options.flow[Z]) id = Z;

  cs = inflow->cs;
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);
  cs_nhalo(cs, &nhalo);

  if (noffset[id] == 0) {

    /* The inflow region covers the halo only in the flow direction. */

    int imin[3] = {1, 1, 1};
    int imax[3] = {nlocal[X], nlocal[Y], nlocal[Z]};

    imin[id] = 1 - nhalo;
    imax[id] = 0;

    for (int ic = imin[X]; ic <= imax[X]; ic++) {
      for (int jc = imin[Y]; jc <= imax[Y]; jc++) {
	for (int kc = imin[Z]; kc <= imax[Z]; kc++) {
	  int index = cs_index(inflow->cs,ic, jc, kc);
	  field_scalar_set(phi, index, inflow->options.phi0);
	}
      }
    }
  }

  return 0;
}
