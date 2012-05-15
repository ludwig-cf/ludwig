/*****************************************************************************
 *
 *  psi_colloid.c
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "util.h"
#include "coords.h"
#include "psi.h"
#include "colloids.h"

/*****************************************************************************
 *
 *  psi_colloid_rho_set
 *
 *  Distribute the total charge of the colloid to the lattice sites
 *  depending on the current discrete volume.
 *
 *  It is envisaged that this is called before a halo exchange of
 *  the lattice psi->rho values. However, the process undertaken
 *  here could be extended into the halo regions if a halo swap
 *  is not required for other reasons.
 *
 *  A more effiecient version would probably compute and store
 *  1/volume for each particle in advance before trawling through
 *  the lattice.
 *
 *****************************************************************************/

int psi_colloid_rho_set(psi_t * obj) {

  int ic, jc, kc, index;
  int nlocal[3];

  double rho0, rho1, volume;
  colloid_t * pc = NULL;
  colloid_t * colloid_at_site_index(int index);

  assert(obj);

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	pc = colloid_at_site_index(index);

	if (pc) {
	  util_discrete_volume_sphere(pc->s.r, pc->s.a0, &volume);
	  rho0 = pc->s.q0/volume;
	  rho1 = pc->s.q1/volume;
	  psi_rho_set(obj, index, 0, rho0);
	  psi_rho_set(obj, index, 1, rho1);
	}

	/* Next site */
      }
    }
  }

  return 0;
}
