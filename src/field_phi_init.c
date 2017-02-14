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
