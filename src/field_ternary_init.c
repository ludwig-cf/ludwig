/****************************************************************************
 *
 *  field_ternary_init.c
 *
 *  Initial configurations intended for ternary mixtures.
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
#include "field_phi_init.h"

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
  double phi0, psi0, phipsi[2];
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

        phi0 = 0.0;                
	psi0 = 1.0;

	if (x < 0.3*len[X]) {
	  phi0 = +1.0;
	  psi0 =  0.0;
	}
	if (x > 0.6*len[X]) {
	  phi0 = -1.0;
	  psi0 =  0.0;
	}

	phipsi[0] = phi0;
	phipsi[1] = psi0;
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
 *  Initialise a "double emulsion":
 *
 *       phi = 0
 *     -------------------
 *     |        |        |
 *     |  +1    |  -1    |
 *     -------------------
 *
 *  cf. Semprebon et al (2016) Fig 1 (e).
 *
 *****************************************************************************/

int field_ternary_init_2d_double_emulsion(field_t * phi) {
    
  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double x, y;
  double phi0, psi0, phipsi[3];
  double len[3];

  double x1, x2, x3;
  double y1, y2;

  assert(phi);
  
  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);

  /* Block positions */
  x1 = 0.2*len[X];
  x2 = 0.5*len[X];
  x3 = 0.8*len[X];
  y1 = 0.3*len[Y];
  y2 = 0.7*len[Y];

  /* rho = 1 */
  phipsi[2] = 1.0;


  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	
	index = cs_index(phi->cs, ic, jc, kc);
	x = noffset[X] + ic;
	y = noffset[X] + jc;

	/* Compoenent c3 */
	phi0 = 0.0;
	psi0 = 1.0;

	if (x1 < x && x < x2  &&  y1 < y  && y < y2) {
	  /* Component c1 */
	  phi0 = +1.0;
	  psi0 =  0.0;
	}
	if (x2 < x && x < x3  &&  y1 < y  && y < y2) {
	  /* Compoenent c2 */
	  phi0 = -1.0;
	  psi0 =  0.0;
	}

	phipsi[0] = phi0;
	phipsi[1] = psi0;
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
 *       |      |     |     areas 1/3 : 1/3 : 1/3
 *       | c1   | c2  |
 *       |------------|
 *       |    c3      |
 *       +------------+
 *
 *  cf. Semprebon etal (2016) Fig. 2 (a)
 *
 *****************************************************************************/

int field_ternary_init_2d_tee(field_t * phi) {
    
  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double x, y;
  double phi0, psi0, phipsi[3];
  double len[3];

  double x1, y1;

  assert(phi);
  
  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);

  /* Block positions */
  x1 = 0.50*len[X];
  y1 = 0.33*len[Y];

  /* rho = 1 */
  phipsi[2] = 1.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	
	index = cs_index(phi->cs, ic, jc, kc);
	x = noffset[X] + ic;
	y = noffset[X] + jc;

	if (y < y1) {
	  /* Compoenent c3 */
	  phi0 = 0.0;
	  psi0 = 1.0;
	}
	else {
	  if (x < x1) {
	    /* Component c1 */
	    phi0 = +1.0;
	    psi0 =  0.0;
	  }
	  else {
	    /* Compoenent c2 */
	    phi0 = -1.0;
	    psi0 =  0.0;
	  }
	}

	phipsi[0] = phi0;
	phipsi[1] = psi0;
	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }
    
  return 0;
}

/*****************************************************************************
 *
 *****************************************************************************/

int field_ternary_init_bbb(field_t * phi) {
    
  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double x, y;
  double phi0, psi0, phipsi[2];
  double len[3];

  double x0, y0, r;

  assert(phi);

  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);

  x0 = 0.50*len[X];
  y0 = 0.50*len[Y];
  r  = 0.25*len[Y];

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= 0.5*nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
                
	index = cs_index(phi->cs, ic, jc, kc);
	x = noffset[X] + ic;
	y = noffset[X] + jc;

	phi0 = 1.0;
	psi0 = 0.0;
	if ((x-x0)*(x-x0) + (y-y0)*(y-y0) <= r*r) {
	  phi0 = 0.0;
	  psi0 = 1.0;
	}

	phipsi[0] = phi0;
	phipsi[1] = psi0;
	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 0.5*nlocal[Y]; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
                
	index = cs_index(phi->cs, ic, jc, kc);
	x = noffset[X] + ic;
	y = noffset[X] + jc;

	phi0 = -1.0;
	psi0 =  0.0;
	if ((x-x0)*(x-x0) + (y-y0)*(y-y0) <= r*r) {
	  phi0 = 0.0;
	  psi0 = 1.0;
	}

	phipsi[0] = phi0;
	phipsi[1] = psi0;
	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  field_ternary_init_ggg
 *
 *  Initialise one block of chosen thickness at central position on X
 *
 *****************************************************************************/

int field_ternary_init_ggg(field_t * phi) {
    
  int nlocal[3];
  int noffset[3];
  int ic, jc, kc, index;
  double x, y;
  double phi0, psi0, phipsi[2];
  double len[3];

  double xl, xr, y0, r;
    
  assert(phi);

  cs_nlocal(phi->cs, nlocal);
  cs_nlocal_offset(phi->cs, noffset);
  cs_ltot(phi->cs, len);
  
  xl = 0.36*len[X];
  xr = 0.64*len[X];
  y0 = 0.50*len[Y];
  r  = 0.15*len[X];

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
                
	index = cs_index(phi->cs, ic, jc, kc);
	x = noffset[X] + ic;
	y = noffset[X] + jc;
                
        phi0 = 0.0;
	psi0 = 1.0;
	if ((x-xl)*(x-xl) + (y-y0)*(y-y0) <= r*r) {
	  phi0 =  1.0;
	  psi0 =  0.0;
	}
        if ((x-xr)*(x-xr) + (y-y0)*(y-y0) <= r*r) {
	  phi0 = -1.0;
	  psi0 =  0.0;
	}

	phipsi[0] = phi0;
	phipsi[1] = psi0;
	field_scalar_array_set(phi, index, phipsi);
      }
    }
  }

  return 0;
}
