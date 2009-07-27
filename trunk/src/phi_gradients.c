/*****************************************************************************
 *
 *  phi_gradients.c
 *
 *  Compute various gradients in the order parameter.
 *
 *  $Id: phi_gradients.c,v 1.8 2009-07-27 08:58:31 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "site_map.h"
#include "free_energy.h"
#include "leesedwards.h"
#include "phi.h"
#include "phi_gradients.h"

extern double * phi_site;
extern double * delsq_phi_site;
extern double * grad_phi_site;
extern double * delsq_delsq_phi_site;
extern double * grad_delsq_phi_site;

/* These are the 'links' used to form the gradients of the order
 * parameter at solid/fluid boundaries. The order could be
 * optimised */

#define NGRAD_ 27
static const int bs_cv[NGRAD_][3] = {{ 0, 0, 0},
				 {-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},
                                 {-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},
                                 {-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},
                                 { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},
                                 { 0, 0,-1},             { 0, 0, 1},
				 { 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},
				 { 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},
				 { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},
				 { 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}};


static void phi_gradients_with_solid(void);
static void phi_gradients_walls(void);
static void phi_gradients_fluid(void);
static void phi_gradients_leesedwards(void);
static void (* phi_gradient_function)(void) = phi_gradients_fluid;
static void f_grad_phi(int, int, int, int, int, const int *);
static void f_delsq_phi(int, int, int, int, int, const int *);

static void phi_gradients_fluid_inline(void);
static void phi_gradients_double_fluid_inline(void);

/****************************************************************************
 *
 *  phi_gradients_set_fluid
 *
 *  Set the gradient calculation for fluid only.
 *
 ****************************************************************************/

void phi_gradients_set_fluid() {

  phi_gradient_function = phi_gradients_fluid;
  return;
}

/****************************************************************************
 *
 *  phi_gradients_set_solid() {
 *
 *  Set the gradient calculation to allow for solids.
 *
 ****************************************************************************/

void phi_gradients_set_solid() {

  phi_gradient_function = phi_gradients_with_solid;
  return;
}

/****************************************************************************
 *
 *  phi_gradients_compute
 *
 *  Depending on the free energy, we may need gradients of the order
 *  parameter up to a certain order...
 *
 ****************************************************************************/

void phi_gradients_compute() {

  assert(phi_gradient_function);

  phi_leesedwards_transformation();
  phi_gradient_function();
  phi_gradients_leesedwards();

  /* Non-periodic x-direction requires corrections */

  if (!is_periodic(X)) phi_gradients_walls();

  /* Brazovskii requires gradients up to nabla^2(\nabla^2) phi */
  /* There also needs to be the appropriate correction if LE is required */

  if (free_energy_is_brazovskii()) phi_gradients_double_fluid();

  return;
}

/****************************************************************************
 *
 *  phi_gradients_with_solid
 *
 *  Compute the gradients of the phi field. This is the 'predictor
 *  corrector' method described by Desplat et al. Comp. Phys. Comm.
 *  134, 273--290 (2000) to take account of solid objects.
 *
 *  This calculation can be extended into the halo region by
 *  (nhalo_ - 1) points in each direction.
 *
 ****************************************************************************/

static void phi_gradients_with_solid() {

  int nlocal[3];
  int ic, jc, kc, ic1, jc1, kc1;
  int ia, index, n, p;
  int nextra = nhalo_ - 1; /* Gradients not computed at last point locally */

  int isite[NGRAD_];
  double count[NGRAD_];
  double gradt[NGRAD_];
  double gradn[3];
  double dphi;
  double rk = 1.0/free_energy_K();
  const double r9 = (1.0/9.0);     /* normaliser for cv_bs */
  const double r18 = (1.0/18.0);   /* ditto */

  get_N_local(nlocal);
  assert(nhalo_ >= 1);
  assert(le_get_nplane_total() == 0);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = le_site_index(ic, jc, kc);
	if (site_map_get_status(ic, jc, kc) != FLUID) continue;

	/* Set solid/fluid flag to index neighbours */

	for (p = 1; p < NGRAD_; p++) {
	  ic1 = ic + bs_cv[p][X];
	  jc1 = jc + bs_cv[p][Y];
	  kc1 = kc + bs_cv[p][Z];

	  isite[p] = le_site_index(ic1, jc1, kc1);
	  if (site_map_get_status(ic1, jc1, kc1) != FLUID) isite[p] = -1;
	}

	for (n = 0; n < nop_; n++) {

	  for (ia = 0; ia < 3; ia++) {
	    count[ia] = 0.0;
	    gradn[ia] = 0.0;
	  }
	  
	  for (p = 1; p < NGRAD_; p++) {

	    if (isite[p] == -1) continue;
	    dphi = phi_site[nop_*isite[p] + n] - phi_site[nop_*index + n];
	    gradt[p] = dphi;

	    for (ia = 0; ia < 3; ia++) {
	      gradn[ia] += bs_cv[p][ia]*dphi;
	      count[ia] += bs_cv[p][ia]*bs_cv[p][ia];
	    }
	  }

	  for (ia = 0; ia < 3; ia++) {
	    if (count[ia] > 0.0) gradn[ia] /= count[ia];
	  }

	  /* Estimate gradient at boundaries */

	  for (p = 1; p < NGRAD_; p++) {

	    if (isite[p] == -1) {
	      double c, h, phi_b;
	      phi_b = phi_site[nop_*index + n]
		+ 0.5*(bs_cv[p][X]*gradn[X] + bs_cv[p][Y]*gradn[Y]
		       + bs_cv[p][Z]*gradn[Z]);

	      /* Set gradient phi at boundary following wetting properties */
	      /* C is always zero at the moment */

	      ia = le_site_index(ic + bs_cv[p][X], jc + bs_cv[p][Y],
				 kc + bs_cv[p][Z]);
	      c = 0.0; /* could be site_map_get_C() */
	      h = site_map_get_H(ia);

	      /* kludge: if nop_ is 2, set h[1] = 0 */
	      h = (1 - n)*h;

	      gradt[p] = -(c*phi_b + h)*rk;
	    }
	  }
 
	  /* Accumulate the final gradients */

	  dphi = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    gradn[ia] = 0.0;
	  }

	  for (p = 1; p < NGRAD_; p++) {
	    dphi += gradt[p];
	    for (ia = 0; ia < 3; ia++) {
	      gradn[ia] += gradt[p]*bs_cv[p][ia];
	    }
	  }

	  delsq_phi_site[nop_*index + n] = r9*dphi;
	  for (ia = 0; ia < 3; ia++) {
	    grad_phi_site[3*(nop_*index + n) + ia]  = r18*gradn[ia];
	  }
	}

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_walls
 *
 *  This is a special case where solid walls are required at x = 0
 *  and x = L_x to run with the Lees-Edwards planes. It is called
 *  after phi_gradients_fluid to correct the gradients at the
 *  edges of the system.
 *
 *  This is done in preference to phi_gradients_solid, which is much
 *  slower. Always neutral wetting.
 *
 *****************************************************************************/

static void phi_gradients_walls() {

  int nlocal[3];
  int ic, jc, kc;
  int ia, index, index1, n, p;
  int nextra = nhalo_ - 1; /* Gradients not computed at last point locally */

  double gradt[NGRAD_];
  double gradn[3];
  double dphi;
  const double r9 = (1.0/9.0);     /* normaliser for cv_bs */
  const double r18 = (1.0/18.0);   /* ditto */

  get_N_local(nlocal);
  assert(nhalo_ >= 1);

  if (cart_coords(X) == 0) {
    ic = 1;

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = le_site_index(ic, jc, kc);

	for (n = 0; n < nop_; n++) {
	  
	  /* This loop is hardwired to pick up bs_cv[p][X] >= 0
	   * without a conditional */
	  for (p = 1; p < 10; p++) {
	    gradt[p] = 0.0;
	  }
	  for (p = 10; p < NGRAD_; p++) {
	    index1 = le_site_index(ic + bs_cv[p][X], jc + bs_cv[p][Y],
				   kc + bs_cv[p][Z]);
	    dphi = phi_site[nop_*index1 + n] - phi_site[nop_*index + n];
	    gradt[p] = dphi;
	  }
 
	  /* Accumulate the final gradients */

	  dphi = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    gradn[ia] = 0.0;
	  }

	  for (p = 1; p < NGRAD_; p++) {
	    dphi += gradt[p];
	    for (ia = 0; ia < 3; ia++) {
	      gradn[ia] += gradt[p]*bs_cv[p][ia];
	    }
	  }

	  delsq_phi_site[nop_*index + n] = r9*dphi;
	  for (ia = 0; ia < 3; ia++) {
	    grad_phi_site[3*(nop_*index + n) + ia]  = r18*gradn[ia];
	  }
	}
	/* Next site */
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {
    ic = nlocal[X];

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = le_site_index(ic, jc, kc);

	for (n = 0; n < nop_; n++) {

	  /* Again, need to pick up only bs_cv[p][X] <= 0 */
	  for (p = 1; p < 18; p++) {
	    index1 = le_site_index(ic + bs_cv[p][X], jc + bs_cv[p][Y],
				   kc + bs_cv[p][Z]);
	    dphi = phi_site[nop_*index1 + n] - phi_site[nop_*index + n];
	    gradt[p] = dphi;
	  }
	  for (p = 18; p < NGRAD_; p++) {
	    gradt[p] = 0.0;
	  }
 
	  /* Accumulate the final gradients */

	  dphi = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    gradn[ia] = 0.0;
	  }

	  for (p = 1; p < NGRAD_; p++) {
	    dphi += gradt[p];
	    for (ia = 0; ia < 3; ia++) {
	      gradn[ia] += gradt[p]*bs_cv[p][ia];
	    }
	  }

	  delsq_phi_site[nop_*index + n] = r9*dphi;
	  for (ia = 0; ia < 3; ia++) {
	    grad_phi_site[3*(nop_*index + n) + ia]  = r18*gradn[ia];
	  }
	}
	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_fluid
 *
 *  Fluid-only gradient calculation. This is an unrolled version.
 *  It is much faster than the compact version.
 *
 *****************************************************************************/

static void phi_gradients_fluid() {

  int nlocal[3];
  int ic, jc, kc;
  int icp1, icm1;
  int nextra = nhalo_ - 1;

  get_N_local(nlocal);
  assert(nhalo_ >= 1);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	f_grad_phi(icm1, ic, icp1, jc, kc, nlocal);
	f_delsq_phi(icm1, ic, icp1, jc, kc, nlocal);

	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_double_fluid
 *
 *  Computes higher derivatives:
 *    \nabla (\nabla^2 phi)
 *    \nabla^2 \nabla^2 phi
 *
 *  This calculation can be extended (nhalo_ - 2) points beyond the
 *  local system.
 *
 *  Brazovskii: at the moment this will operate successfully to
 *  produce the higher derivatives in the domain proper for
 *  nhalo_ = 2, i.e., ok using the relaxation toward the chemical
 *  stress to force the two-distribution LB approach. Can't use
 *  finite difference yet as this would require the extra
 *  derivatives in the halo / Lees Edwards buffer regions.
 *
 *****************************************************************************/
 
void phi_gradients_double_fluid() {

  int nlocal[3];
  int ic, jc, kc, ic1, jc1, kc1;
  int ia, index, index1, p;
  int nextra = nhalo_ - 2;

  double phi0, phi1;
  const double r9 = (1.0/9.0);
  const double r18 = (1.0/18.0);

  get_N_local(nlocal);
  assert(nhalo_ >= 2);
  assert(nop_ == 1);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = le_site_index(ic, jc, kc);
	phi0 = delsq_phi_site[index];

	for (ia = 0; ia < 3; ia++) {
	  grad_delsq_phi_site[3*index + ia] = 0.0;
	}
	delsq_delsq_phi_site[index] = 0.0;

	for (p = 1; p < NGRAD_; p++) {
	  ic1 = le_index_real_to_buffer(ic, bs_cv[p][X]);
	  jc1 = jc + bs_cv[p][Y];
	  kc1 = kc + bs_cv[p][Z];
	  index1 = ADDR(ic1, jc1, kc1);
	  phi1 = delsq_phi_site[index1];

	  for (ia = 0; ia < 3; ia++) {
	    grad_delsq_phi_site[3*index + ia] += r18*bs_cv[p][ia]*phi1;
	  }
	  delsq_delsq_phi_site[index] += r9*(phi1 - phi0);
	}
 
	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_leesedwards
 *
 *  The gradients of the order parameter need to be set in the
 *  buffer region to the extent of (nhalo_ - 1) planes either
 *  side of the Lees Edwards boundary.
 *
 *****************************************************************************/

static void phi_gradients_leesedwards() {

  int nlocal[3];
  int n, nh;
  int nextra = nhalo_ - 1;
  int ic, ic0, ic1, ic2, jc, kc;

  get_N_local(nlocal);

  for (n = 0; n < le_get_nplane_local(); n++) {

    ic = le_plane_location(n);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(ic, nh-1);
      ic1 = le_index_real_to_buffer(ic, nh  );
      ic2 = le_index_real_to_buffer(ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {
	  f_grad_phi(ic0, ic1, ic2, jc, kc, nlocal);
	  f_delsq_phi(ic0, ic1, ic2, jc, kc, nlocal);
	}
      }

    }

    /* Looking across the plane in the -ve x-direction. */
    ic += 1;

    for (nh = 1; nh <= nextra; nh++) {
      ic2 = le_index_real_to_buffer(ic, -nh+1);
      ic1 = le_index_real_to_buffer(ic, -nh  );
      ic0 = le_index_real_to_buffer(ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {
	  f_grad_phi(ic0, ic1, ic2, jc, kc, nlocal);
	  f_delsq_phi(ic0, ic1, ic2, jc, kc, nlocal);
	}
      }
    }
    /* Next plane */
  }

  return;
}

/*****************************************************************************
 *
 *  f_grad_phi
 *
 *  Sets grad_phi at (ic,jc,kc) from differences of phi_site[]. The
 *  explicit im, ic, ip, allow the buffer regions to be used when
 *  Lees Edwards planes are present.
 *
 *****************************************************************************/

static void f_grad_phi(int icm1, int ic, int icp1, int jc, int kc,
		       const int nlocal[3]) {

  const double r18 = 1.0/18.0;
  int n;

  for (n = 0; n < nop_; n++) {

    grad_phi_site[3*(nop_*ADDR(ic,jc,kc) + n) + X] =
      r18*(phi_site[nop_*ADDR(icp1,jc,  kc  ) + n]
	 - phi_site[nop_*ADDR(icm1,jc,  kc  ) + n]
	 + phi_site[nop_*ADDR(icp1,jc+1,kc+1) + n]
	 - phi_site[nop_*ADDR(icm1,jc+1,kc+1) + n]
	 + phi_site[nop_*ADDR(icp1,jc-1,kc+1) + n]
	 - phi_site[nop_*ADDR(icm1,jc-1,kc+1) + n]
	 + phi_site[nop_*ADDR(icp1,jc+1,kc-1) + n]
	 - phi_site[nop_*ADDR(icm1,jc+1,kc-1) + n]
	 + phi_site[nop_*ADDR(icp1,jc-1,kc-1) + n]
	 - phi_site[nop_*ADDR(icm1,jc-1,kc-1) + n]
	 + phi_site[nop_*ADDR(icp1,jc+1,kc  ) + n]
	 - phi_site[nop_*ADDR(icm1,jc+1,kc  ) + n]
	 + phi_site[nop_*ADDR(icp1,jc-1,kc  ) + n]
	 - phi_site[nop_*ADDR(icm1,jc-1,kc  ) + n]
	 + phi_site[nop_*ADDR(icp1,jc,  kc+1) + n]
	 - phi_site[nop_*ADDR(icm1,jc,  kc+1) + n]
	 + phi_site[nop_*ADDR(icp1,jc,  kc-1) + n]
	 - phi_site[nop_*ADDR(icm1,jc,  kc-1) + n]);
		    
    grad_phi_site[3*(nop_*ADDR(ic,jc,kc) + n) + Y] = 
      r18*(phi_site[nop_*ADDR(ic  ,jc+1,kc  ) + n]
	   - phi_site[nop_*ADDR(ic,  jc-1,kc  ) + n]
	   + phi_site[nop_*ADDR(icp1,jc+1,kc+1) + n]
	   - phi_site[nop_*ADDR(icp1,jc-1,kc+1) + n]
	   + phi_site[nop_*ADDR(icm1,jc+1,kc+1) + n]
	   - phi_site[nop_*ADDR(icm1,jc-1,kc+1) + n]
	   + phi_site[nop_*ADDR(icp1,jc+1,kc-1) + n]
	   - phi_site[nop_*ADDR(icp1,jc-1,kc-1) + n]
	   + phi_site[nop_*ADDR(icm1,jc+1,kc-1) + n]
	   - phi_site[nop_*ADDR(icm1,jc-1,kc-1) + n]
	   + phi_site[nop_*ADDR(icp1,jc+1,kc  ) + n]
	   - phi_site[nop_*ADDR(icp1,jc-1,kc  ) + n]
	   + phi_site[nop_*ADDR(icm1,jc+1,kc  ) + n]
	   - phi_site[nop_*ADDR(icm1,jc-1,kc  ) + n]
	   + phi_site[nop_*ADDR(ic,  jc+1,kc+1) + n]
	   - phi_site[nop_*ADDR(ic,  jc-1,kc+1) + n]
	   + phi_site[nop_*ADDR(ic,  jc+1,kc-1) + n]
	   - phi_site[nop_*ADDR(ic,  jc-1,kc-1) + n]);
		    
    grad_phi_site[3*(nop_*ADDR(ic,jc,kc) + n) + Z] = 
      r18*(phi_site[nop_*ADDR(ic,  jc,  kc+1) + n]
	   - phi_site[nop_*ADDR(ic,  jc,  kc-1) + n]
	   + phi_site[nop_*ADDR(icp1,jc+1,kc+1) + n]
	   - phi_site[nop_*ADDR(icp1,jc+1,kc-1) + n]
	   + phi_site[nop_*ADDR(icm1,jc+1,kc+1) + n]
	   - phi_site[nop_*ADDR(icm1,jc+1,kc-1) + n]
	   + phi_site[nop_*ADDR(icp1,jc-1,kc+1) + n]
	   - phi_site[nop_*ADDR(icp1,jc-1,kc-1) + n]
	   + phi_site[nop_*ADDR(icm1,jc-1,kc+1) + n]
	   - phi_site[nop_*ADDR(icm1,jc-1,kc-1) + n]
	   + phi_site[nop_*ADDR(icp1,jc,  kc+1) + n]
	   - phi_site[nop_*ADDR(icp1,jc,  kc-1) + n]
	   + phi_site[nop_*ADDR(icm1,jc,  kc+1) + n]
	   - phi_site[nop_*ADDR(icm1,jc,  kc-1) + n]
	   + phi_site[nop_*ADDR(ic,  jc+1,kc+1) + n]
	   - phi_site[nop_*ADDR(ic,  jc+1,kc-1) + n]
	   + phi_site[nop_*ADDR(ic,  jc-1,kc+1) + n]
	   - phi_site[nop_*ADDR(ic,  jc-1,kc-1) + n]);
  }

  return;
}

/*****************************************************************************
 *
 *  f_delsq_phi
 *
 *  Finite difference for delsq_phi.
 *
 *****************************************************************************/

static void f_delsq_phi(int icm1, int ic, int icp1, int jc, int kc,
			const int nlocal[3]) {

  const double r9 = 1.0/9.0;
  int n;

  for (n = 0; n < nop_; n++) {

    delsq_phi_site[nop_*ADDR(ic,jc,kc) + n] =
      r9*(phi_site[nop_*ADDR(icp1,jc,  kc  ) + n]
	  + phi_site[nop_*ADDR(icm1,jc,  kc  ) + n]
	  + phi_site[nop_*ADDR(ic,  jc+1,kc  ) + n]
	  + phi_site[nop_*ADDR(ic,  jc-1,kc  ) + n]
	  + phi_site[nop_*ADDR(ic,  jc,  kc+1) + n]
	  + phi_site[nop_*ADDR(ic,  jc,  kc-1) + n]
	  + phi_site[nop_*ADDR(icp1,jc+1,kc+1) + n]
	  + phi_site[nop_*ADDR(icp1,jc+1,kc-1) + n]
	  + phi_site[nop_*ADDR(icp1,jc-1,kc+1) + n]
	  + phi_site[nop_*ADDR(icp1,jc-1,kc-1) + n]
	  + phi_site[nop_*ADDR(icm1,jc+1,kc+1) + n]
	  + phi_site[nop_*ADDR(icm1,jc+1,kc-1) + n]
	  + phi_site[nop_*ADDR(icm1,jc-1,kc+1) + n]
	  + phi_site[nop_*ADDR(icm1,jc-1,kc-1) + n]
	  + phi_site[nop_*ADDR(icp1,jc+1,kc  ) + n]
	  + phi_site[nop_*ADDR(icp1,jc-1,kc  ) + n]
	  + phi_site[nop_*ADDR(icm1,jc+1,kc  ) + n]
	  + phi_site[nop_*ADDR(icm1,jc-1,kc  ) + n]
	  + phi_site[nop_*ADDR(icp1,jc,  kc+1) + n]
	  + phi_site[nop_*ADDR(icp1,jc,  kc-1) + n]
	  + phi_site[nop_*ADDR(icm1,jc,  kc+1) + n]
	  + phi_site[nop_*ADDR(icm1,jc,  kc-1) + n]
	  + phi_site[nop_*ADDR(ic,  jc+1,kc+1) + n]
	  + phi_site[nop_*ADDR(ic,  jc+1,kc-1) + n]
	  + phi_site[nop_*ADDR(ic,  jc-1,kc+1) + n]
	  + phi_site[nop_*ADDR(ic,  jc-1,kc-1) + n]
	  - 26.0*phi_site[nop_*ADDR(ic,jc,kc) + n]);
  }

  return;
}

/****************************************************************************
 *
 *  phi_gradients_fluid_inline
 *
 ****************************************************************************/

static void phi_gradients_fluid_inline(void) {

  int nlocal[3];
  int ic, jc, kc;
  int index;
  int nextra = nhalo_ - 1;
  int xs, ys;
 	 
  double phi0;
  const double r9 = (1.0/9.0);
  const double r18 = (1.0/18.0);
 	 
  get_N_local(nlocal);
  assert(nhalo_ >= 1);
  assert(nop_ == 1);
  assert(le_get_nplane_total() == 0);

  /* Strides in x- and y-directions */
  xs = (nlocal[Y] + 2*nhalo_)*(nlocal[Z] + 2*nhalo_);
  ys = (nlocal[Z] + 2*nhalo_);
 	 
  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {
 	 
	index = get_site_index(ic, jc, kc);
	phi0 = phi_site[index];
 	 
	grad_phi_site[3*index + X] =
	  r18*(phi_site[index+xs     ]-phi_site[index-xs     ] +
	       phi_site[index+xs+ys+1]-phi_site[index-xs+ys+1] +
	       phi_site[index+xs-ys+1]-phi_site[index-xs-ys+1] +
	       phi_site[index+xs+ys-1]-phi_site[index-xs+ys-1] +
	       phi_site[index+xs-ys-1]-phi_site[index-xs-ys-1] +
	       phi_site[index+xs+ys  ]-phi_site[index-xs+ys  ] +
	       phi_site[index+xs-ys  ]-phi_site[index-xs-ys  ] +
	       phi_site[index+xs   +1]-phi_site[index-xs   +1] +
	       phi_site[index+xs   -1]-phi_site[index-xs   -1]);
 	                     
	grad_phi_site[3*index + Y] =
	  r18*(phi_site[index   +ys  ]-phi_site[index   -ys  ] +
	       phi_site[index+xs+ys+1]-phi_site[index+xs-ys+1] +
	       phi_site[index-xs+ys+1]-phi_site[index-xs-ys+1] +
	       phi_site[index+xs+ys-1]-phi_site[index+xs-ys-1] +
	       phi_site[index-xs+ys-1]-phi_site[index-xs-ys-1] +
	       phi_site[index+xs+ys  ]-phi_site[index+xs-ys  ] +
	       phi_site[index-xs+ys  ]-phi_site[index-xs-ys  ] +
	       phi_site[index   +ys+1]-phi_site[index   -ys+1] +
	       phi_site[index   +ys-1]-phi_site[index   -ys-1]);
 	                     
        grad_phi_site[3*index + Z] =
	  r18*(phi_site[index      +1]-phi_site[index      -1] +
	       phi_site[index+xs+ys+1]-phi_site[index+xs+ys-1] +
	       phi_site[index-xs+ys+1]-phi_site[index-xs+ys-1] +
	       phi_site[index+xs-ys+1]-phi_site[index+xs-ys-1] +
	       phi_site[index-xs-ys+1]-phi_site[index-xs-ys-1] +
	       phi_site[index+xs   +1]-phi_site[index+xs   -1] +
	       phi_site[index-xs   +1]-phi_site[index-xs   -1] +
	       phi_site[index   +ys+1]-phi_site[index   +ys-1] +
	       phi_site[index   -ys+1]-phi_site[index   -ys-1]);
 	                     
	delsq_phi_site[index] = r9*(phi_site[index+xs     ] +
				    phi_site[index-xs     ] +
				    phi_site[index   +ys  ] +
				    phi_site[index   -ys  ] +
				    phi_site[index      +1] +
				    phi_site[index      -1] +
				    phi_site[index+xs+ys+1] +
				    phi_site[index+xs+ys-1] +
				    phi_site[index+xs-ys+1] +
				    phi_site[index+xs-ys-1] +
				    phi_site[index-xs+ys+1] +
				    phi_site[index-xs+ys-1] +
				    phi_site[index-xs-ys+1] +
				    phi_site[index-xs-ys-1] +
				    phi_site[index+xs+ys  ] +
				    phi_site[index+xs-ys  ] +
				    phi_site[index-xs+ys  ] +
				    phi_site[index-xs-ys  ] +
				    phi_site[index+xs   +1] +
				    phi_site[index+xs   -1] +
				    phi_site[index-xs   +1] +
				    phi_site[index-xs   -1] +
				    phi_site[index   +ys+1] +
				    phi_site[index   +ys-1] +
				    phi_site[index   -ys+1] +
				    phi_site[index   -ys-1] -
				    26.0*phi0);
	/* Next site */
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  phi_gradients_double_fluid_inline
 *
 ****************************************************************************/

static void phi_gradients_double_fluid_inline(void) {

  int nlocal[3];
  int ic, jc, kc;
  int index;
  int nextra = nhalo_ - 1;
  int xs, ys;                   
 	 
  double phi0;
  const double r9 = (1.0/9.0);
  const double r18 = (1.0/18.0);
 	 
  get_N_local(nlocal);
  assert(nhalo_ >= 1);
  assert(nop_ == 1);
  assert(le_get_nplane_total() == 0);

  /* Stride in the x- and y-directions */
  xs = (nlocal[Y] + 2*nhalo_)*(nlocal[Z] + 2*nhalo_);
  ys = (nlocal[Z] + 2*nhalo_);
 	 
  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {
 	 
	index = get_site_index(ic, jc, kc);
	phi0 = delsq_phi_site[index];
 	 
	grad_delsq_phi_site[3*index + X] =
	  r18*(delsq_phi_site[index+xs     ]-delsq_phi_site[index-xs     ] +
	       delsq_phi_site[index+xs+ys+1]-delsq_phi_site[index-xs+ys+1] +
	       delsq_phi_site[index+xs-ys+1]-delsq_phi_site[index-xs-ys+1] +
	       delsq_phi_site[index+xs+ys-1]-delsq_phi_site[index-xs+ys-1] +
	       delsq_phi_site[index+xs-ys-1]-delsq_phi_site[index-xs-ys-1] +
	       delsq_phi_site[index+xs+ys  ]-delsq_phi_site[index-xs+ys  ] +
	       delsq_phi_site[index+xs-ys  ]-delsq_phi_site[index-xs-ys  ] +
	       delsq_phi_site[index+xs   +1]-delsq_phi_site[index-xs   +1] +
	       delsq_phi_site[index+xs   -1]-delsq_phi_site[index-xs   -1]);
 	                     
	grad_delsq_phi_site[3*index + Y] =
	  r18*(delsq_phi_site[index   +ys  ]-delsq_phi_site[index   -ys  ] +
	       delsq_phi_site[index+xs+ys+1]-delsq_phi_site[index+xs-ys+1] +
	       delsq_phi_site[index-xs+ys+1]-delsq_phi_site[index-xs-ys+1] +
	       delsq_phi_site[index+xs+ys-1]-delsq_phi_site[index+xs-ys-1] +
	       delsq_phi_site[index-xs+ys-1]-delsq_phi_site[index-xs-ys-1] +
	       delsq_phi_site[index+xs+ys  ]-delsq_phi_site[index+xs-ys  ] +
	       delsq_phi_site[index-xs+ys  ]-delsq_phi_site[index-xs-ys  ] +
	       delsq_phi_site[index   +ys+1]-delsq_phi_site[index   -ys+1] +
	       delsq_phi_site[index   +ys-1]-delsq_phi_site[index   -ys-1]);
 	                     
        grad_delsq_phi_site[3*index + Z] =
	  r18*(delsq_phi_site[index      +1]-delsq_phi_site[index      -1] +
	       delsq_phi_site[index+xs+ys+1]-delsq_phi_site[index+xs+ys-1] +
	       delsq_phi_site[index-xs+ys+1]-delsq_phi_site[index-xs+ys-1] +
	       delsq_phi_site[index+xs-ys+1]-delsq_phi_site[index+xs-ys-1] +
	       delsq_phi_site[index-xs-ys+1]-delsq_phi_site[index-xs-ys-1] +
	       delsq_phi_site[index+xs   +1]-delsq_phi_site[index+xs   -1] +
	       delsq_phi_site[index-xs   +1]-delsq_phi_site[index-xs   -1] +
	       delsq_phi_site[index   +ys+1]-delsq_phi_site[index   +ys-1] +
	       delsq_phi_site[index   -ys+1]-delsq_phi_site[index   -ys-1]);
 	                     
	delsq_delsq_phi_site[index] = r9*(delsq_phi_site[index+xs     ] +
					  delsq_phi_site[index-xs     ] +
					  delsq_phi_site[index   +ys  ] +
					  delsq_phi_site[index   -ys  ] +
					  delsq_phi_site[index      +1] +
					  delsq_phi_site[index      -1] +
					  delsq_phi_site[index+xs+ys+1] +
					  delsq_phi_site[index+xs+ys-1] +
					  delsq_phi_site[index+xs-ys+1] +
					  delsq_phi_site[index+xs-ys-1] +
					  delsq_phi_site[index-xs+ys+1] +
					  delsq_phi_site[index-xs+ys-1] +
					  delsq_phi_site[index-xs-ys+1] +
					  delsq_phi_site[index-xs-ys-1] +
					  delsq_phi_site[index+xs+ys  ] +
					  delsq_phi_site[index+xs-ys  ] +
					  delsq_phi_site[index-xs+ys  ] +
					  delsq_phi_site[index-xs-ys  ] +
					  delsq_phi_site[index+xs   +1] +
					  delsq_phi_site[index+xs   -1] +
					  delsq_phi_site[index-xs   +1] +
					  delsq_phi_site[index-xs   -1] +
					  delsq_phi_site[index   +ys+1] +
					  delsq_phi_site[index   +ys-1] +
					  delsq_phi_site[index   -ys+1] +
					  delsq_phi_site[index   -ys-1] -
					  26.0*phi0);
	/* Next site */
      }
    }
  }

  return;
}

