/*****************************************************************************
 *
 *  phi_gradients.c
 *
 *  Compute various gradients in the order parameter.
 *
 *  $Id: phi_gradients.c,v 1.10.4.4 2010-03-30 14:29:47 kevin Exp $
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
#include "phi.h"
#include "gradient.h"
#include "phi_gradients.h"

extern double * phi_site;
extern double * delsq_phi_site;
extern double * grad_phi_site;
extern double * delsq_delsq_phi_site;
extern double * grad_delsq_phi_site;

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

static int      dyadic_required_ = 0;
static double * phi_delsq_pp_;
static double * phi_dpp_;

static void phi_gradients_walls(void);

/****************************************************************************
 *
 *  phi_gradients_init
 *
 ****************************************************************************/

void phi_gradients_init(void) {

  int nop;
  int nsites;

  nop = phi_nop();
  nsites = coords_nsites();

  if (dyadic_required_) {
    assert(nop == 3);
    phi_delsq_pp_ = (double *) malloc(18*nop*nsites*sizeof(double));
    phi_dpp_ = (double *) malloc(6*nop*nsites*sizeof(double));

    if (phi_delsq_pp_ == NULL) fatal("malloc(phi_delsq_pp_) failed\n");
    if (phi_dpp_ == NULL) fatal("malloc(phi_dpp_) failed\n");
  }

  return;
}

/****************************************************************************
 *
 *  phi_gradients_finish
 *
 ****************************************************************************/

void phi_gradients_finish(void) {

  if (phi_delsq_pp_) free(phi_delsq_pp_);
  if (phi_dpp_) free(phi_dpp_);

  return;
}

/****************************************************************************
 *
 *  phi_gradients_compute
 *
 *  To compute the gradients, the order parameter must be translated
 *  to take account of the Lees Edwards sliding blocks. The gradients
 *  can then be comupted as required.
 *
 ****************************************************************************/

void phi_gradients_compute() {

  int nop;

  nop = phi_nop();

  if (0) {
    extern void gradient_2d_5pt_fluid_d2(void);

    phi_leesedwards_transformation();
    gradient_2d_5pt_fluid_d2();
    gradient_2d_5pt_fluid_d4();
    phi_solid_walls();
  }
  else {

  phi_leesedwards_transformation();

  gradient_d2(nop, phi_site, grad_phi_site, delsq_phi_site);

  if (phi_gradient_level() > 2) {
    gradient_d4(nop, delsq_phi_site, grad_delsq_phi_site,
		delsq_delsq_phi_site);
  }

  if (dyadic_required_) {
    gradient_d2_dyadic(nop, phi_site, phi_dpp_, phi_delsq_pp_);
  }

  /* Remaining to test */
  /* phi_gradients_walls();*/

  }
  return;
}


/*****************************************************************************
 *
 *  phi_gradients_grad_dyadic
 *
 *  Return d_c q_a q_b for vector order parameter.
 *
 *****************************************************************************/

void phi_gradients_grad_dyadic(const int index, double dqq[3][3][3]) {

  int ia;

  assert(phi_dpp_);

  for (ia = 0; ia < 3; ia++) {
    dqq[ia][X][X] = phi_dpp_[18*index + 6*ia + XX];
    dqq[ia][X][Y] = phi_dpp_[18*index + 6*ia + XY];
    dqq[ia][X][Z] = phi_dpp_[18*index + 6*ia + XZ];
    dqq[ia][Y][X] = dqq[X][X][Y];
    dqq[ia][Y][Y] = phi_dpp_[18*index + 6*ia + YY];
    dqq[ia][Y][Z] = phi_dpp_[18*index + 6*ia + YZ];
    dqq[ia][Z][X] = dqq[X][X][Z];
    dqq[ia][Z][Y] = dqq[X][Y][Z];
    dqq[ia][Z][Z] = phi_dpp_[18*index + 6*ia + ZZ];
  }

  return;
}

/*****************************************************************************
 *
 *  phi_gradients_delsq_dyadic
 *
 *  Return nabla^2 q_a q_b for vector order parameter q_a
 *
 *****************************************************************************/

void phi_gradients_delsq_dyadic(const int index, double delsq[3][3]) {

  assert(phi_delsq_pp_);

  delsq[X][X] = phi_delsq_pp_[6*index + XX];
  delsq[X][Y] = phi_delsq_pp_[6*index + XY];
  delsq[X][Z] = phi_delsq_pp_[6*index + XZ];
  delsq[Y][X] = delsq[X][Y];
  delsq[Y][Y] = phi_delsq_pp_[6*index + YY];
  delsq[Y][Z] = phi_delsq_pp_[6*index + YZ];
  delsq[Z][X] = delsq[X][Z];
  delsq[Z][Y] = delsq[Y][Z];
  delsq[Z][Z] = phi_delsq_pp_[6*index + ZZ];

  return;
}

/* OLD STUFF TO GO */

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

	index = coords_index(ic, jc, kc);

	for (n = 0; n < nop_; n++) {
	  
	  /* This loop is hardwired to pick up bs_cv[p][X] >= 0
	   * without a conditional */
	  for (p = 1; p < 10; p++) {
	    gradt[p] = 0.0;
	  }
	  for (p = 10; p < NGRAD_; p++) {
	    index1 = coords_index(ic + bs_cv[p][X], jc + bs_cv[p][Y],
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

	index = coords_index(ic, jc, kc);

	for (n = 0; n < nop_; n++) {

	  /* Again, need to pick up only bs_cv[p][X] <= 0 */
	  for (p = 1; p < 18; p++) {
	    index1 = coords_index(ic + bs_cv[p][X], jc + bs_cv[p][Y],
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
