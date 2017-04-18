/****************************************************************************
 *
 *  field_phi_init.c
 *
 *  Initial compositional order parameter configurations.
 *  Independent of the free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "noise.h"
#include "util.h"
#include "field_phi_init.h"

/*****************************************************************************
 *
 *  field_phi_init_drop
 *
 *  Droplet based on a profile phi(r) = phistar tanh (r-r0)/xi
 *  with r0 the centre of the system.
 *
 *****************************************************************************/

int field_phi_init_drop(double xi, double radius,
			double phistar, field_t * phi) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
  double centre[3];
  double phival, r, rxi0;

  assert(phi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  rxi0 = 1.0/xi;

  centre[X] = 0.5*L(X);
  centre[Y] = 0.5*L(Y);
  centre[Z] = 0.5*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
        position[X] = 1.0*(noffset[X] + ic) - centre[X];
        position[Y] = 1.0*(noffset[Y] + jc) - centre[Y];
        position[Z] = 1.0*(noffset[Z] + kc) - centre[Z];

        r = sqrt(dot_product(position, position));

        phival = phistar*tanh(rxi0*(r - radius));
	field_scalar_set(phi, index, phival);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * field_phi_init_block
 *
 *  Initialise two blocks with interfaces at z = Lz/4 and z = 3Lz/4.
 *
 *****************************************************************************/

int field_phi_init_block(double xi, field_t * phi) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z1, z2;
  double phi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  z1 = 0.25*L(Z);
  z2 = 0.75*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	z = noffset[Z] + kc;

	if (z > 0.5*L(Z)) {
	  phi0 = tanh((z-z2)/xi);
	}
	else {
	  phi0 = -tanh((z-z1)/xi);
	}

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * field_phi_init_block_X
 *
 *  Initialise one block of chosen thickness at central position on X
 *
 *****************************************************************************/

int field_phi_init_block_X(double xi, field_t * phi, double block_dimension) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double x, x1, x2;
  double phi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  x1 = 0.5*(L(X)-block_dimension);
  x2 = x1+block_dimension;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	x = noffset[X] + ic;

	if (x > 0.5*L(X)) {
	  phi0 = tanh((x-x2)/xi);
	}
	else {
	  phi0 = -tanh((x-x1)/xi);
	}

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * field_phi_init_block_Y
 *
 *  Initialise one block of chosen thickness at central position on Y
 *
 *****************************************************************************/

int field_phi_init_block_Y(double xi, field_t * phi, double block_dimension) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double y, y1, y2;
  double phi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  y1 = 0.5*(L(Y)-block_dimension);
  y2 = y1+block_dimension;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	y = noffset[Y] + jc;

	if (y > 0.5*L(Y)) {
	  phi0 = tanh((y-y2)/xi);
	}
	else {
	  phi0 = -tanh((y-y1)/xi);
	}

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * field_phi_init_block_Z
 *
 *  Initialise one block of chosen thickness at central position on Z
 *
 *****************************************************************************/

int field_phi_init_block_Z(double xi, field_t * phi, double block_dimension) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z1, z2;
  double phi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  z1 = 0.5*(L(Z)-block_dimension);
  z2 = z1+block_dimension;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	z = noffset[Z] + kc;

	if (z > 0.5*L(Z)) {
	  phi0 = tanh((z-z2)/xi);
	}
	else {
	  phi0 = -tanh((z-z1)/xi);
	}

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * field_phi_init_layer_X
 *
 *  Initialise two layers with an interface at chosen position on X
 *
 *****************************************************************************/

int field_phi_init_layer_X(double xi, field_t * phi, double layer_size) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double x, x1;
  double phi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  x1 = layer_size*L(X);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	x = noffset[X] + ic;
        phi0 = tanh((x-x1)/xi);

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * field_phi_init_layer_Y
 *
 *  Initialise two layers with an interface at chosen position on Y
 *
 *****************************************************************************/

int field_phi_init_layer_Y(double xi, field_t * phi, double layer_size) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double y, y1;
  double phi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  y1 = layer_size*L(Y);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	y = noffset[Y] + jc;
        phi0 = tanh((y-y1)/xi);

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 * field_phi_init_layer_Z
 *
 *  Initialise two layers with an interface at chosen position on Z
 *
 *****************************************************************************/

int field_phi_init_layer_Z(double xi, field_t * phi, double layer_size) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z1;
  double phi0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  z1 = layer_size*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	z = noffset[Z] + kc;
        phi0 = tanh((z-z1)/xi);

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_phi_init_bath
 *
 *  Initialise one interface at z = Lz/8. This is inended for
 *  capillary rise in systems with z not periodic.
 *
 *****************************************************************************/

int field_phi_init_bath(field_t * phi) {

  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double z, z0;
  double phi0, xi0;

  assert(phi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  z0 = 0.25*L(Z);
  xi0 = 1.13;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) { 
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	z = noffset[Z] + kc;
	phi0 = tanh((z-z0)/xi0);

	field_scalar_set(phi, index, phi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_phi_init_spinodal
 *
 *  Some random noise is required to initiate spinodal decomposition.
 *
 *  For different random numbers r based on seed, at each point:
 *  phi = phi0 + amp*(r - 0.5) with r uniform on [0,1).
 *
 *****************************************************************************/

int field_phi_init_spinodal(int seed, double phi0, double amp,
			    field_t * phi) {

  int ic, jc, kc, index;
  int nlocal[3];

  double ran;
  double phi1;

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  noise_t * rng = NULL;

  assert(phi);

  pe_ref(&pe);
  cs_ref(&cs);
  coords_nlocal(nlocal);

  noise_create(pe, cs, &rng);
  noise_init(rng, seed);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	noise_uniform_double_reap(rng, index, &ran);
	phi1 = phi0 + amp*(ran - 0.5);
	field_scalar_set(phi, index, phi1);

      }
    }
  }

  noise_free(rng);

  return 0;
}

/*****************************************************************************
 *
 *  field_phi_init_spinodal_patches
 *
 *  Also for spinodal, but a slightly different strategy of patches,
 *  which is better for large composition ratios.
 *
 *  Generally, the further away from 50:50 one moves, the larger
 *  the patch size must be to prevent diffusion (via the mobility)
 *  washing out the spinodal decomposition.
 *
 *  Composition is initialised with phi = +1 or phi = -1
 *  Patch is the patch size in lattice sites
 *  volminus1 is the overall fraction of the phi = -1 phase.
 *
 *****************************************************************************/

int field_phi_init_spinodal_patches(int seed, int patch,
				    double volminus1, field_t * phi) {

  int ic, jc, kc, index;
  int ip, jp, kp;
  int nlocal[3];
  int ipatch, jpatch, kpatch;
  int count = 0;

  double phi1;
  double ran_uniform;

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  noise_t * rng = NULL;

  assert(phi);

  pe_ref(&pe);
  cs_ref(&cs);
  coords_nlocal(nlocal);

  noise_create(pe, cs, &rng);
  noise_init(rng, seed);

  for (ic = 1; ic <= nlocal[X]; ic += patch) {
    for (jc = 1; jc <= nlocal[Y]; jc += patch) {
      for (kc = 1; kc <= nlocal[Z]; kc += patch) {

	index = coords_index(ic, jc, kc);

	/* Uniform patch */
	phi1 = 1.0;
	noise_uniform_double_reap(rng, index, &ran_uniform);
	if (ran_uniform < volminus1) phi1 = -1.0;

	ipatch = dmin(nlocal[X], ic + patch - 1);
	jpatch = dmin(nlocal[Y], jc + patch - 1);
	kpatch = dmin(nlocal[Z], kc + patch - 1);

	for (ip = ic; ip <= ipatch; ip++) {
	  for (jp = jc; jp <= jpatch; jp++) {
	    for (kp = kc; kp <= kpatch; kp++) {

	      index = coords_index(ip, jp, kp);
	      field_scalar_set(phi, index, phi1);
	      count += 1;
	    }
	  }
	}

	/* Next patch */
      }
    }
  }

  noise_free(rng);

  assert(count == nlocal[X]*nlocal[Y]*nlocal[Z]);

  return 0;
}

/*****************************************************************************
 *
 *  field_phi_init_emulsion
 *
 *  Initialises the binary OP as an array of droplets.
 *  The system considered here is 2D, where X << Y,Z.
 *
 *****************************************************************************/

int field_phi_init_emulsion(double xi, double radius, double phistar, int N_drops, 
     double d_centre, field_t * phi) {

  int ntotal[3];
  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;

  int i, j, k, x, y, z;
  int id, drop_temp;
  double dy, dz;
  int cy, cz;

  int PosY[N_drops], PosZ[N_drops];
  int PosY0, PosZ0;
  double Rclosest; 
  int Nclosest;

  double r;
  int ny, nz; // number of drops on each dimension
  double rxi0; // Interface width
  double phi1; /* Final phi value */

  assert(phi);

  coords_ntotal(ntotal);
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  double Phi[ntotal[0]][ntotal[1]][ntotal[2]];

  info("Starting function symmetric_init_emulsion\n");

  rxi0 = 1.0/xi;

  if (ntotal[Z]>=ntotal[Y]) {
    ny = ceil(sqrt(ntotal[Z]*ntotal[Y]*N_drops)/(ntotal[Z]*1.0) );
    nz = ceil(ny*ntotal[Z]/ntotal[Y]);
  }
  else {
    nz = ceil(sqrt(ntotal[Y]*ntotal[Z]*N_drops)/(ntotal[Y]*1.0) );
    ny = ceil(nz*ntotal[Y]/ntotal[Z]);
  }

  info("Maximum number of drops on y and z: %i  %i\n", ny, nz);

  PosY0 = (int)(d_centre/2.0);
  PosZ0 = (int)(d_centre/2.0);

  info("Position of the first drop %i %i\n", PosY0, PosZ0);

  cy=1;
  cz=1;
  dy = d_centre;
  dz = sqrt(dy*dy - d_centre*d_centre/4.0);

  for (id=0; id<N_drops; id++) {

    if (cz/2 != (1.0*cz)/2.0) {
      PosY[id] = (int)(PosY0 + (cy-1.0)*dy);
      PosZ[id] = (int)(PosZ0 + (cz-1.0)*dz);
    }
    else {
      PosY[id] = (int)(PosY0 + dy/2.0 + (cy-1.0)*dy);
      PosZ[id] = (int)(PosZ0 + (cz-1.0)*dz);
    }

    cy++;

    if (PosY[id]+d_centre/2.0+d_centre > ntotal[Y]) {
      cz++;
      cy=1;
    }
    info("Position of drop %i -> %i %i\n", id, PosY[id], PosZ[id]);

  }

  for (i=0; i<ntotal[X]; i++) {
    for (j=0; j<ntotal[Y]; j++) {
      for (k=0; k<ntotal[Z]; k++) {

	drop_temp = 0;
	Phi[i][j][k] = -(phistar);

	for (id=0; id<N_drops; id++) {

	  r = sqrt((j-PosY[id])*(j-PosY[id]) + (k-PosZ[id])*(k-PosZ[id]));
	  if (r <= d_centre/2.0) {
	    Phi[i][j][k] = -1.0*phistar*tanh(rxi0*(r-radius));
	  }
	  if (r <= radius) {
	    drop_temp = id+1; 
	  }

	} 

      }
    }
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	x = noffset[X] + ic;
	y = noffset[Y] + jc;
	z = noffset[Z] + kc;

	phi1 = -(phistar);
	Nclosest = 0;
	Rclosest = ntotal[Y]*ntotal[Z];

	for (id=0; id<N_drops; id++) {

	  r = sqrt((y-PosY[id])*(y-PosY[id]) + (z-PosZ[id])*(z-PosZ[id]));
	  if (r < Rclosest) {
	    Nclosest = id;
	    Rclosest = r;
	  }

	}

	phi1 = -1.0*phistar*tanh(rxi0*(Rclosest-radius));

	field_scalar_set(phi, index, phi1);

      }
    }
  }

  return 0;
}

