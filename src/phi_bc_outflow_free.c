/*****************************************************************************
 *
 *  phi_bc_outflow_free.c
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "phi_bc_outflow_free.h"

static const phi_bc_open_vtable_t vt_ = {
  (phi_bc_open_free_ft)   phi_bc_outflow_free_free,
  (phi_bc_open_update_ft) phi_bc_outflow_free_update
};

/*****************************************************************************
 *
 *  phi_bc_outflow_free_create
 *
 *****************************************************************************/

__host__ int phi_bc_outflow_free_create(pe_t * pe, cs_t * cs,
					 const phi_bc_outflow_opts_t * options,
					 phi_bc_outflow_free_t ** outflow) {
  phi_bc_outflow_free_t * bc = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(outflow);

  bc = (phi_bc_outflow_free_t *) calloc(1, sizeof(phi_bc_outflow_free_t));
  assert(bc);
  if (bc == NULL) pe_fatal(pe, "calloc phi_bc_outflow_free_t failed\n");

  /* Pointers */
  bc->pe = pe;
  bc->cs = cs;

  if (phi_bc_outflow_opts_valid(*options) == 0) {
    pe_fatal(pe, "Internal error: phi_bc_outflow_opts not valid\n");
  }
  bc->options = *options;

  bc->super.func = &vt_;
  bc->super.id   = PHI_BC_OUTFLOW_FREE;

  *outflow = bc;

  return 0;
}

/*****************************************************************************
 *
 *  phi_bc_outflow_free_free
 *
 *****************************************************************************/

__host__ int phi_bc_outflow_free_free(phi_bc_outflow_free_t * outflow) {

  assert(outflow);

  free(outflow);

  return 0;
}

/*****************************************************************************
 *
 *  phi_bc_outflow_free_update
 *
 *  Important assumption here. We should have up-to-date halo information
 *  in the non-flow direction, so that we can push the update into the
 *  non-flow halo directions locally.
 *
 *  For complete safety, there should probably be an in-built halo
 *  exchange to ensure the relevant information is up-to-date.
 *
 *****************************************************************************/

__host__ int phi_bc_outflow_free_update(phi_bc_outflow_free_t * outflow,
					field_t * phi) {
  int id = -1;
  int nhalo = -1;
  int nlocal[3] = {};
  int ntotal[3] = {};
  int noffset[3] = {};
  cs_t * cs = NULL;

  assert(outflow);
  assert(phi);

  if (outflow->options.flow[X] == 1) id = X;
  if (outflow->options.flow[Y] == 1) id = Y;
  if (outflow->options.flow[Z] == 1) id = Z;

  cs = outflow->cs;
  cs_ntotal(cs, ntotal);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);
  cs_nhalo(cs, &nhalo);

  if (noffset[id] + nlocal[id] == ntotal[id]) {
    /* Set halo region e.g., phi(x > L_x, y, z) = phi(x = L_x, y, z) */
    /* Slightly convoluted to allow x, y, or z direction. */

    int imin[3] = {1-nhalo, 1-nhalo, 1-nhalo};
    int imax[3] = {nlocal[X]+nhalo, nlocal[Y]+nhalo, nlocal[Z]+nhalo};

    imin[id] = nlocal[id];
    imax[id] = nlocal[id];

    for (int ic = imin[X]; ic <= imax[X]; ic++) {
      for (int jc = imin[Y]; jc <= imax[Y]; jc++) {
	for (int kc = imin[Z]; kc <= imax[Z]; kc++) {
	  int index = cs_index(cs, ic, jc, kc);
	  double phi0 = 0.0;
	  field_scalar(phi, index, &phi0);  /* Domain value */
	  for (int nh = 1; nh <= nhalo; nh++) {
	    int di[3] = {nh*outflow->options.flow[X],
	                 nh*outflow->options.flow[Y],
			 nh*outflow->options.flow[Z]};
	    index = cs_index(cs, ic + di[X], jc + di[Y], kc + di[Z]);
	    field_scalar_set(phi, index, phi0);
	  }
	}
      }
    }
  }

  return 0;
}
