/****************************************************************************
 *
 *  field_ternary_init.c
 *
 *  Initial configurations intended for ternary mixtures following
 *  Semprebon etal (2016).
 *
 *  In general, we have values of phi, psi:
 *
 *     phase   rho   phi   psi
 *     -----------------------
 *     one       1    +1     0
 *     two       1    -1     0
 *     three     1     0    +1
 *
 *  where c_1 = (rho + phi - psi)/2, c_2 = (rho - phi - psi)/2, c_3 = psi.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "noise.h"
#include "util.h"
#include "field_s.h"
#include "field_ternary_init.h"

/* Three phases c_1, c_2, c_3 */
#define COMPONENT1 1
#define COMPONENT2 2
#define COMPONENT3 3

/*****************************************************************************
 *
 *  field_ternary_init_phase
 *
 *  Helper to initialise a triple (phi, psi, rho) from "phase 1, 2, 3".
 *
 *****************************************************************************/

int field_ternary_init_phase(int iphase, double scalar[3]) {

  const int FE_PHI = 0; /* To be consistent with free energy! */
  const int FE_PSI = 1;
  const int FE_RHO = 2;
  int ierr = 0;

  switch (iphase) {
  case COMPONENT1:
    scalar[FE_RHO] =  1.0;
    scalar[FE_PHI] = +1.0;
    scalar[FE_PSI] =  0.0;
    break;
  case COMPONENT2:
    scalar[FE_RHO] =  1.0;
    scalar[FE_PHI] = -1.0;
    scalar[FE_PSI] =  0.0;
    break;
  case COMPONENT3:
    scalar[FE_RHO] =  1.0;
    scalar[FE_PHI] =  0.0;
    scalar[FE_PSI] =  1.0;
    break;
  default:
    assert(0);
    ierr = -1;
  }

  return ierr;
}

/*****************************************************************************
 *
 *  field_ternary_init_X
 *
 *****************************************************************************/

int field_ternary_init_X(field_t * phi) {
    
  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double x;
  double phipsi[3];
  double len[3];

  assert(phi);
    
  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);
    
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
                
	index = cs_index(phi->cs, ic, jc, kc);
	x = noffset[X] + ic;

	field_ternary_init_phase(COMPONENT3, phipsi);

	if (x < 0.3*len[X]) {
	  field_ternary_init_phase(COMPONENT1, phipsi);
	}
	if (x > 0.6*len[X]) {
	  field_ternary_init_phase(COMPONENT2, phipsi);
	}

	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }
    
  return 0;
}

/*****************************************************************************
 *
 *  field_ternary_init_2d_double_emulsion
 *
 *  Initialise a "double emulsion" as two central blocks in (x,y,z=1):
 *
 *       c_3
 *     -------------------     y2
 *     |        |        |
 *     |  c_1   |  c_2   |
 *     -------------------     y1
 *
 *     x1       x2       x3
 *
 *  Phase 3, c_3, is then continuous at the outer boundaries.
 *  cf. the result of Semprebon et al (2016) Fig 1 (e).
 *
 *****************************************************************************/

int field_ternary_init_2d_double_emulsion(field_t * phi,
					  const fti_block_t * param) {
  int nlocal[3];
  int noffset[3];
  double phipsi[3];
  double len[3];

  double x1, x2, x3;
  double y1, y2;

  assert(phi);
  assert(param);

  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);

  /* Block positions */

  x1 = param->xf1*len[X];
  x2 = param->xf2*len[X];
  x3 = param->xf3*len[X];
  y1 = param->yf1*len[Y];
  y2 = param->yf2*len[Y];

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(phi->cs, ic, jc, kc);
	double x = noffset[X] + ic;
	double y = noffset[X] + jc;

	/* Component c3 is default */
	field_ternary_init_phase(COMPONENT3, phipsi);

	if (x1 <  x && x < x2  &&  y1 < y  && y < y2) {
	  /* Component c1 */
	  field_ternary_init_phase(COMPONENT1, phipsi);
	}
	if (x2 <= x && x < x3  &&  y1 < y  && y < y2) {
	  /* Compoenent c2 */
	  field_ternary_init_phase(COMPONENT2, phipsi);
	}

	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_ternary_init_2d_tee
 *
 *  Initialise three phases in a "T" shape:
 *
 *       +------------+     There should be roughly equal
 *       |      |     |     areas 1/3 : 1/3 : 1/3 in default parameters.
 *       | c1   | c2  |
 *       |------------|
 *       |    c3      |
 *       +------------+
 *
 *  There are two parameters involved (xf1,yf1) for vertical and horizontal
 *  cuts, respectively. This should give rise to a situation analogous to
 *  Semprebon etal (2016) Fig. 2 (a) with wetting.
 *
 *****************************************************************************/

int field_ternary_init_2d_tee(field_t * phi, const fti_block_t * param) {
    
  int nlocal[3];
  int noffset[3];
  double phipsi[3];
  double len[3];

  double x1, y1;

  assert(phi);
  assert(param);
  
  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);

  /* Block positions */

  x1 = param->xf1*len[X];
  y1 = param->yf1*len[Y];

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	
	int index = cs_index(phi->cs, ic, jc, kc);
	double x = noffset[X] + ic;
	double y = noffset[X] + jc;

	if (y < y1) {
	  /* Compoenent c3 */
	  field_ternary_init_phase(COMPONENT3, phipsi);
	}
	else {
	  if (x < x1) {
	    /* Component c1 */
	    field_ternary_init_phase(COMPONENT1, phipsi);
	  }
	  else {
	    /* Compoenent c2 */
	    field_ternary_init_phase(COMPONENT2, phipsi);
	  }
	}

	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }
    
  return 0;
}

/*****************************************************************************
 *
 *  field_ternary_init_2d_lens
 *
 *  This gives rise to a droplet in a split system. Schematically;
 *
 *     +-------------+
 *     | c_2 __      |    Drop centre (x0, y0) radius r; component c_3
 *     |____/  \_____|
 *     |    \__/     |    Horizontal cut always at Ly/2
 *     | c_1         |
 *     +-------------+
 *
 *  This should give rise to the lens-like configuration seen in
 *  Semprebon etal (2016) Fig. 1 (b) or (d) depending on the
 *  equilibrium contact angles.
 *
 *****************************************************************************/

int field_ternary_init_2d_lens(field_t * phi, const fti_drop_t * drop) {
    
  int nlocal[3];
  int noffset[3];
  double phipsi[3];
  double len[3];

  double x0, y0, r;

  assert(phi);

  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);

  x0 = drop->r0[X];
  y0 = drop->r0[Y];
  r  = drop->r;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
                
	int index = cs_index(phi->cs, ic, jc, kc);
	double x = noffset[X] + ic;
	double y = noffset[Y] + jc;

	/* Horizontal cut */
	if (y < 0.5*len[Y]) {
	  field_ternary_init_phase(COMPONENT1, phipsi);
	}
	else {
	  field_ternary_init_phase(COMPONENT2, phipsi);
	}

	/* Drop is superposed... */
	if ((x-x0)*(x-x0) + (y-y0)*(y-y0) < r*r) {
	  field_ternary_init_phase(COMPONENT3, phipsi);
	}

	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_ternary_init_2d_double_drop
 *
 *  Two drops (left and right, or 1 and 2) with separate (r0, r).
 *  Drop 1 is component 1, drop two is component 2.
 *  Background is component 3.
 *
 *  A double emulsion.
 *
 *****************************************************************************/

int field_ternary_init_2d_double_drop(field_t * phi, const fti_drop_t * drop1,
				      const fti_drop_t * drop2) {
    
  int nlocal[3];
  int noffset[3];
  double phipsi[3];
  double len[3];
    
  assert(phi);
  assert(drop1);
  assert(drop2);

  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
                
	int index = cs_index(phi->cs, ic, jc, kc);
	double x = noffset[X] + ic;
	double y = noffset[X] + jc;

	/* Background */
	field_ternary_init_phase(COMPONENT3, phipsi);

	{
	  /* Drop 1 */
	  double x0 = drop1->r0[X];
	  double y0 = drop1->r0[Y];
	  double r  = drop1->r;
	  if ((x-x0)*(x-x0) + (y-y0)*(y-y0) < r*r) {
	    field_ternary_init_phase(COMPONENT1, phipsi);
	  }
	}

	{
	  /* Drop 2 (may superpose drop 1) */
	  double x0 = drop2->r0[X];
	  double y0 = drop2->r0[Y];
	  double r  = drop2->r;
	  if ((x-x0)*(x-x0) + (y-y0)*(y-y0) < r*r) {
	    field_ternary_init_phase(COMPONENT2, phipsi);
	  }
	}

	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }

  return 0;
}
