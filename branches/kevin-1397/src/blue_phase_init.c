/*****************************************************************************
 *
 *  blue_phase_init.c
 *
 *  Various initial states for liquid crystal blue phases.
 *
 *  See, for example, Henrich et al, ...
 *  and references therein.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (o.henrich@ucl.ac.uk)
 *  Juho Lintuvuori
 *
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "field.h"
#include "field_grad.h"
#include "blue_phase.h"
#include "blue_phase_init.h"
#include "blue_phase_beris_edwards.h"
#include "ran.h"
#include "noise.h"

#define DEFAULT_SEED 13

void blue_phase_M_rot(double M[3][3], int dim, double alpha);

/*****************************************************************************
 *
 *  blue_phase_O8M_init
 *
 *  BP I using the current free energy parameter q0
 *
 *****************************************************************************/

int blue_phase_O8M_init(fe_lc_param_t * param, field_t * fq) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double r2;
  double cosx, cosy, cosz, sinx, siny, sinz;
  double q0;
  double amplitude0;

  assert(fq);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  r2 = sqrt(2.0);
  q0 = param->q0;
  amplitude0 = param->amplitude0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	cosx = cos(r2*q0*x);
	cosy = cos(r2*q0*y);
	cosz = cos(r2*q0*z);
	sinx = sin(r2*q0*x);
	siny = sin(r2*q0*y);
	sinz = sin(r2*q0*z);

	q[X][X] = amplitude0*(-2.0*cosy*sinz +    sinx*cosz + cosx*siny);
	q[X][Y] = amplitude0*(  r2*cosy*cosz + r2*sinx*sinz - sinx*cosy);
	q[X][Z] = amplitude0*(  r2*cosx*cosy + r2*sinz*siny - cosx*sinz);
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude0*(-2.0*sinx*cosz +    siny*cosx + cosy*sinz);
	q[Y][Z] = amplitude0*(  r2*cosz*cosx + r2*siny*sinx - siny*cosz);
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	field_tensor_set(fq, index, q);

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_O2_init
 *
 *  This initialisation is for BP II.
 *
 *****************************************************************************/

int blue_phase_O2_init(fe_lc_param_t * param, field_t * fq) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double q0;
  double amplitude0;

  assert(fq);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  q0 = param->q0;
  amplitude0 = param->amplitude0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude0*(cos(2.0*q0*z) - cos(2.0*q0*y));
	q[X][Y] = amplitude0*sin(2.0*q0*z);
	q[X][Z] = amplitude0*sin(2.0*q0*y);
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude0*(cos(2.0*q0*x) - cos(2.0*q0*z));
	q[Y][Z] = amplitude0*sin(2.0*q0*x);
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_H2D_init
 *
 *  This initialisation is for 2D hexagonal BP.
 *
 *****************************************************************************/

int blue_phase_H2D_init(fe_lc_param_t * param, field_t * fq) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y;
  double r3;
  double q0;
  double amplitude0;

  assert(fq);

  r3 = sqrt(3.0);
  q0 = param->q0;
  amplitude0 = param->amplitude0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude0*(-1.5*   cos(q0*x)*cos(q0*r3*y));
	q[X][Y] = amplitude0*(-0.5*r3*sin(q0*x)*sin(q0*r3*y));
	q[X][Z] = amplitude0*(     r3*cos(q0*x)*sin(q0*r3*y));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude0*(-cos(2.0*q0*x) - 0.5*cos(q0*x)*cos(q0*r3*y));
	q[Y][Z] = amplitude0*(-sin(2.0*q0*x) -     sin(q0*x)*cos(q0*r3*y));
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_H3DA_init
 *
 *  This initialisation is for 3D hexagonal BP A.
 *
 *****************************************************************************/

int blue_phase_H3DA_init(fe_lc_param_t * param, field_t * fq) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double r3;
  double q0;
  double amplitude0;

  assert(fq);

  r3 = sqrt(3.0);
  q0 = param->q0;
  amplitude0 = param->amplitude0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude0*(-1.5*cos(q0*x)*cos(q0*r3*y)
			       + 0.25*cos(q0*L(X)/L(Z)*z)); 
	q[X][Y] = amplitude0*(-0.5*r3*sin(q0*x)*sin(q0*r3*y)
			       + 0.25*sin(q0*L(X)/L(Z)*z));
	q[X][Z] = amplitude0*(r3*cos(q0*x)*sin(q0*r3*y));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude0*(-cos(2.0*q0*x)-0.5*cos(q0*x)*cos(q0*r3*y)
			       -0.25*cos(q0*L(X)/L(Z)*z));
	q[Y][Z] = amplitude0*(-sin(2.0*q0*x)-sin(q0*x)*cos(q0*r3*y));
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_H3DB_init
 *
 *  This initialisation is for 3D hexagonal BP B.
 *
 *****************************************************************************/

int blue_phase_H3DB_init(fe_lc_param_t * param, field_t * fq) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double r3;
  double q0;
  double amplitude0;

  assert(fq);

  r3 = sqrt(3.0);
  q0 = param->q0;
  amplitude0 = param->amplitude0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude0*(1.5*cos(q0*x)*cos(q0*r3*y)
			       + 0.25*cos(q0*L(X)/L(Z)*z)); 
	q[X][Y] = amplitude0*(0.5*r3*sin(q0*x)*sin(q0*r3*y)
			       + 0.25*sin(q0*L(X)/L(Z)*z));
	q[X][Z] = amplitude0*(-r3*cos(q0*x)*sin(q0*r3*y));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude0*(cos(2.0*q0*x) + 0.5*cos(q0*x)*cos(q0*r3*y)
			       - 0.25*cos(q0*L(X)/L(Z)*z));
	q[Y][Z] = amplitude0*(sin(2.0*q0*x) + sin(q0*x)*cos(q0*r3*y));
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_O5_init
 *
 *  This initialisation is for O5.
 *
 *****************************************************************************/

int blue_phase_O5_init(fe_lc_param_t * param, field_t * fq) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y, z;
  double q0;
  double amplitude0;

  assert(fq);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  q0 = param->q0;
  amplitude0 = param->amplitude0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	q[X][X] = amplitude0*
            (2.0*cos(sqrt(2.0)*q0*y)*cos(sqrt(2.0)*q0*z)-
                 cos(sqrt(2.0)*q0*x)*cos(sqrt(2.0)*q0*z)-
                 cos(sqrt(2.0)*q0*x)*cos(sqrt(2.0)*q0*y)); 
	q[X][Y] = amplitude0*
            (sqrt(2.0)*cos(sqrt(2.0)*q0*y)*sin(sqrt(2.0)*q0*z)-
             sqrt(2.0)*cos(sqrt(2.0)*q0*x)*sin(sqrt(2.0)*q0*z)-
             sin(sqrt(2.0)*q0*x)*sin(sqrt(2.0)*q0*y));
	q[X][Z] = amplitude0*
            (sqrt(2.0)*cos(sqrt(2.0)*q0*x)*sin(sqrt(2.0)*q0*y)-
             sqrt(2.0)*cos(sqrt(2.0)*q0*z)*sin(sqrt(2.0)*q0*y)-
             sin(sqrt(2.0)*q0*x)*sin(sqrt(2.0)*q0*z));
	q[Y][X] = q[X][Y];
	q[Y][Y] = amplitude0*
            (2.0*cos(sqrt(2.0)*q0*x)*cos(sqrt(2.0)*q0*z)-
                 cos(sqrt(2.0)*q0*y)*cos(sqrt(2.0)*q0*x)-
                 cos(sqrt(2.0)*q0*y)*cos(sqrt(2.0)*q0*z));
	q[Y][Z] = amplitude0*
            (sqrt(2.0)*cos(sqrt(2.0)*q0*z)*sin(sqrt(2.0)*q0*x)-
             sqrt(2.0)*cos(sqrt(2.0)*q0*y)*sin(sqrt(2.0)*q0*x)-
             sin(sqrt(2.0)*q0*y)*sin(sqrt(2.0)*q0*z));
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_DTC_init
 *
 *  This initialisation is with double twist cylinders.
 *
 *****************************************************************************/

int blue_phase_DTC_init(fe_lc_param_t * param, field_t * fq) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, y;
  double q0;
  double amplitude0;

  assert(fq);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  q0 = param->q0;
  amplitude0 = param->amplitude0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	q[X][X] = -amplitude0*cos(2*q0*y);
	q[X][Y] = 0.0;
	q[X][Z] = amplitude0*sin(2.0*q0*y);
	q[Y][X] = q[X][Y];
	q[Y][Y] = -amplitude0*cos(2.0*q0*x);
	q[Y][Z] = -amplitude0*sin(2.0*q0*x);
	q[Z][X] = q[X][Z];
	q[Z][Y] = q[Y][Z];
	q[Z][Z] = - q[X][X] - q[Y][Y];

	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_BPIII_init
 *
 *  This initialisation is with Blue Phase III, randomly positioned
 *  and oriented DTC-cylinders in isotropic (0) or cholesteric (1) environment.
 *
 *  NOTE: The rotations are not rigorously implemented; no cross-boundary 
 *        communication is performed. 
 *        Hence, the decomposition must consist of sufficiently large volumes.
 *        
 *****************************************************************************/

int blue_phase_BPIII_init(fe_lc_param_t * param, field_t * fq, const double specs[3]) {

  int ic, jc, kc;
  int ir, jr, kr; 	/* indices for rotated output */
  int ia, ib, ik, il, in;
  int nlocal[3];
  int noffset[3];
  int index;
  double q[3][3], q0[3][3];
  double x, y, z;
  double *a, *b;	/* rotation angles */
  double *C;     	/* global coordinates of DTC-centres */
  int N=2, R=3, ENV=1; 	/* default no. & radius & environment */ 
  double rc[3];	  	/* distance DTC-centre - site */ 
  double rc_r[3]; 	/* rotated vector */ 
  double Mx[3][3], My[3][3]; /* rotation matrices */
  double phase1, phase2;
  double n[3]={0.0,0.0,0.0};
  double q0_pitch;      /* Just q0 scalar */
  double amplitude0;
  PI_DOUBLE(pi);

  assert(fq);
  assert(specs);

  N = (int) specs[0];
  R = (int) specs[1];
  ENV = (int) specs[2];

  a = (double*)calloc(N, sizeof(double));
  b = (double*)calloc(N, sizeof(double));
  C = (double*)calloc(3*N, sizeof(double));

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  q0_pitch = param->q0;
  amplitude0 = param->amplitude0;

    /* Initialise random rotation angles and centres in serial */
    /* to get the same random numbers on all processes */
    for(in = 0; in < N; in++){

      a[in] = 2.0*pi * ran_serial_uniform();
      b[in] = 2.0*pi * ran_serial_uniform();
      C[3*in]   = N_total(X) * ran_serial_uniform(); 
      C[3*in+1] = N_total(Y) * ran_serial_uniform(); 
      C[3*in+2] = N_total(Z) * ran_serial_uniform(); 

    }

  /* Setting environment configuration */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (ENV == 0){

	  phase1 = pi*(0.5 - ran_parallel_uniform());
	  phase2 = pi*(0.5 - ran_parallel_uniform());

	  n[X] = cos(phase1)*sin(phase2);
	  n[Y] = sin(phase1)*sin(phase2);
	  n[Z] = cos(phase2);

	  fe_lc_q_uniaxial(param, n, q);

	 /* The amplitude of the orderparameter is hardwired */
	 /* for the random and isotropic background configuration */     
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      q[ia][ib] *= 1.0e-6;
	    }
	  }

	}
	if (ENV == 1){

	  /* cholesteric helix along y-direction */
	  n[X] = cos(q0_pitch*y);
	  n[Y] = 0.0;
	  n[Z] = -sin(q0_pitch*y);

	  fe_lc_q_uniaxial(param, n, q);

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      q[ia][ib] *= amplitude0;
	    }
	  }
	  
	}

	index = coords_index(ic, jc, kc);
	field_tensor_set(fq, index, q);
      }
    }
  }

  /* Replace configuration inside DTC-domains */
  /* by sweeping through all local sites */
  for(in = 0; in<N; in++){

    blue_phase_M_rot(Mx,0,a[in]);
    blue_phase_M_rot(My,1,b[in]);

    for (ic = 1; ic <= nlocal[X]; ic++) {
      x = noffset[X] + ic;
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	y = noffset[Y] + jc;
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  z = noffset[Z] + kc;

	  rc[X] = x - C[3*in];
	  rc[Y] = y - C[3*in+1];
	  rc[Z] = z - C[3*in+2];

	  /* If current site is in ROI perform double */
	  /* rotation around local x- and y-axis */
	  if(rc[0]*rc[0] + rc[1]*rc[1] + rc[2]*rc[2] < R*R){

	    for(ia=0; ia<3; ia++){
	      rc_r[ia] = 0.0;
	      for(ik=0; ik<3; ik++){
		for(il=0; il<3; il++){
		  rc_r[ia] += My[ia][ik] * Mx[ik][il] * rc[il];
		}
	      }
	    }

	    /* DTC symmetric wrt local z-axis */
	    q0[X][X] = -amplitude0*cos(2*q0_pitch*rc[Y]);
	    q0[X][Y] = 0.0;
	    q0[X][Z] = amplitude0*sin(2.0*q0_pitch*rc[Y]);
	    q0[Y][X] = q[X][Y];
	    q0[Y][Y] = -amplitude0*cos(2.0*q0_pitch*rc[X]);
	    q0[Y][Z] = -amplitude0*sin(2.0*q0_pitch*rc[X]);
	    q0[Z][X] = q[X][Z];
	    q0[Z][Y] = q[Y][Z];
	    q0[Z][Z] = - q[X][X] - q[Y][Y];

	    /* Transform order parameter tensor */ 
            /* Determine local output index */
            ir = (int)(C[3*in] + rc_r[X] - noffset[X]);
            jr = (int)(C[3*in+1] + rc_r[Y] - noffset[Y]);
            kr = (int)(C[3*in+2] + rc_r[Z] - noffset[Z]);

	    /* Replace if index is in local domain */
	    if((1 <= ir && ir <= nlocal[X]) &&  
	       (1 <= jr && jr <= nlocal[Y]) &&  
               (1 <= kr && kr <= nlocal[Z]))
	    {
   /*******************************************************************
   * NOTE: The individual componentes of the tensor order parameter are
   *       not transformed, i.e. rotated, as this leads to considerable 
   *       instabilities in the calculation of the gradients.
   *       BPIII emerges more reliably from an unrotated OP.
   *******************************************************************/

	      q[X][X] = q0[X][X];
	      q[X][Y] = q0[X][Y];
	      q[X][Z] = q0[X][Z];
	      q[Y][X] = q[X][Y];
	      q[Y][Y] = q0[Y][Y];
	      q[Y][Z] = q0[Y][Z];
	      q[Z][X] = q[X][Z];
	      q[Z][Y] = q[Y][Z];
	      q[Z][Z] = - q[X][X] - q[Y][Y];

	      index = coords_index(ir, jr, kr);
	      field_tensor_set(fq, index, q);
	    }

	  }

	}
      }
    }

  }

  free(a);
  free(b);
  free(C);

  return 0;
}


/*****************************************************************************
 *
 *  blue_phase_twist_init
 *
 *  Initialise a uniaxial helix in the indicated helical axis.
 *  Uses the current free energy parameters
 *     q0 (P=2pi/q0)
 *     amplitude
 *
 *****************************************************************************/

int blue_phase_twist_init(fe_lc_param_t * param, field_t * fq, int helical_axis) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double n[3];
  double q[3][3];
  double x, y, z;
  double q0;

  assert(param);
  assert(fq);
  assert(helical_axis == X || helical_axis == Y || helical_axis == Z);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  q0 = param->q0;

  n[X] = 0.0;
  n[Y] = 0.0;
  n[Z] = 0.0;
 
  for (ic = 1; ic <= nlocal[X]; ic++) {

    if (helical_axis == X) {
      x = noffset[X] + ic;
      n[Y] = cos(q0*x);
      n[Z] = sin(q0*x);
    }

    for (jc = 1; jc <= nlocal[Y]; jc++) {

      if (helical_axis == Y) {
	y = noffset[Y] + jc;
	n[X] = cos(q0*y);
	n[Z] = -sin(q0*y);
      }

      for (kc = 1; kc <= nlocal[Z]; kc++) {
	
	index = coords_index(ic, jc, kc);

	if (helical_axis == Z) {
	  z = noffset[Z] + kc;
	  n[X] = cos(q0*z);
	  n[Y] = sin(q0*z);
	}

	fe_lc_q_uniaxial(param, n, q);
	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_nematic_init
 *
 *  Initialise a uniform uniaxial nematic.
 *
 *  The inputs are the amplitude A and the vector n_a (which we explicitly
 *  convert to a unit vector here).
 *
 *****************************************************************************/

int blue_phase_nematic_init(fe_lc_param_t * param, field_t * fq, const double n[3]) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int ia, index;

  double nhat[3];
  double q[3][3];

  assert(fq);
  assert(n);
  assert(modulus(n) > 0.0);
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ia = 0; ia < 3; ia++) {
    nhat[ia] = n[ia] / modulus(n);
  }

  fe_lc_q_uniaxial(param, nhat, q);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_active_nematic_init
 *
 *  Initialise a uniform uniaxial nematic with a kink in the centre.
 *
 *  This is like the above initialisation in blue_phase_nematic_init
 *  apart from the symmetry-breaking kinks in the middle of the system.
 *
 *****************************************************************************/

int blue_phase_active_nematic_init(fe_lc_param_t * param, field_t * fq, const double n[3]) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int ia, index;

  double nhat[3];
  double q[3][3];

  double nkink1[3], nkink2[3];
  double qkink1[3][3], qkink2[3][3];

  double x, y, z;
  double ang;
  PI_DOUBLE(pi);

  ang = pi/180.0*10.0;

  assert(modulus(n) > 0.0);
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ia = 0; ia < 3; ia++) {
    nhat[ia] = n[ia] / modulus(n);
  }

  /* 2 kinks depending on the primary alignment */

  /* Kink for primary alignment along x */	 
  if (nhat[0] == 1.0) {
    nkink1[0] = nhat[0]*sin(ang);
    nkink1[1] = nhat[1];
    nkink1[2] = nhat[0]*cos(ang);

    nkink2[0] = -nhat[0]*sin(ang);
    nkink2[1] =  nhat[1];
    nkink2[2] =  nhat[0]*cos(ang);
  }
  /* Kink for primary alignment along y */	 

  if (nhat[1] == 1.0) {
    nkink1[0] = nhat[0];
    nkink1[1] = nhat[1]*sin(ang);
    nkink1[2] = nhat[1]*cos(ang);

    nkink2[0] =  nhat[0];
    nkink2[1] = -nhat[1]*sin(ang);
    nkink2[2] =  nhat[1]*cos(ang);
  }

  fe_lc_q_uniaxial(param, nhat, q);
  fe_lc_q_uniaxial(param, nkink1, qkink1);
  fe_lc_q_uniaxial(param, nkink2, qkink2);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);
	field_tensor_set(fq, index, q);

        /* If alignment along x region around 
	   z=N_total(Z)/2 is being replaced */
 
	if(nhat[0] == 1.0) {
	  if(z==N_total(Z)/2.0 || z==(N_total(Z)-1)/2.0) {
	    if(x<=N_total(X)/2.0) {
	      field_tensor_set(fq, index, qkink1);
	    }
	    else {
	      field_tensor_set(fq, index, qkink2);
	    }
	  }
	}

        /* If alignment along y region around 
	   z=N_total(Z)/2 is being replaced */

	if(nhat[1] == 1.0){
	  if(z==N_total(Z)/2.0 || z==(N_total(Z)-1)/2.0) {
	    if(y<=N_total(Y)/2.0) {
	      field_tensor_set(fq, index, qkink1);
	    }
	    else {
	      field_tensor_set(fq, index, qkink2);
	    }
	  }
	}

      }
    }
  }


  return 0;
}

/*****************************************************************************
 *
 *  lc_active_nematic_init_q2d
 *
 *  Intended for quasi-2d systems in which the z-direction is Lz = 1).
 *
 *  For istrip == X, the director looks like
 *
 *       - - - - - -     ^ 
 *       / / / \ \ \     |
 *       - - - - - -     | Y   ---> X
 *
 *  For istrip == Y, swap the labels in the above picture.
 *
 *****************************************************************************/

int lc_active_nematic_init_q2d(fe_lc_param_t * param, field_t * fq,
			       int istrip) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double nhat[3];
  double q[3][3];

  double nkink1[3], nkink2[3];
  double qkink1[3][3], qkink2[3][3];

  double x, y;
  double ang;
  PI_DOUBLE(pi);

  assert(param);
  assert(fq);
  assert(istrip == X || istrip == Y);

  ang = pi/180.0*10.0;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  if (istrip == X) {
    nhat[X] = 1.0;
    nhat[Y] = 0.0;
    nhat[Z] = 0.0;
    nkink1[X] = sin(ang);
    nkink1[Y] = cos(ang);
    nkink1[Z] = 0.0;
    nkink2[X] = -sin(ang);
    nkink2[Y] =  cos(ang);
    nkink2[Z] = 0.0;
  }

  if (istrip == Y) {
    nhat[X] = 0.0;
    nhat[Y] = 1.0;
    nhat[Z] = 0.0;
    nkink1[X] = cos(ang);
    nkink1[Y] = sin(ang);
    nkink1[Z] = 0.0;
    nkink2[X] =  cos(ang);
    nkink2[Y] = -sin(ang);
    nkink2[Z] =  0.0;
  }

  fe_lc_q_uniaxial(param, nhat, q);
  fe_lc_q_uniaxial(param, nkink1, qkink1);
  fe_lc_q_uniaxial(param, nkink2, qkink2);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	/* Background */
	field_tensor_set(fq, index, q);

        /* Central strip parallel to X */
 
	if (istrip == X) {
	  if (y == N_total(Y)/2 || y == (N_total(Y)-1)/2) {
	    if (x <= N_total(X)/2) {
	      field_tensor_set(fq, index, qkink1);
	    }
	    else {
	      field_tensor_set(fq, index, qkink2);
	    }
	  }
	}

        /* Central strip parallel to Y */

	if (istrip == Y) {
	  if (x == N_total(X)/2 || x == (N_total(X)-1)/2) {
	    if (y <= N_total(Y)/2) {
	      field_tensor_set(fq, index, qkink1);
	    }
	    else {
	      field_tensor_set(fq, index, qkink2);
	    }
	  }
	}

	/* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_chi_edge
 *  Setting  chi edge disclination
 *  Using the current free energy parameter q0 (P=2pi/q0)
 *
 *****************************************************************************/

int blue_phase_chi_edge(fe_lc_param_t * param, field_t * fq, int N, double z0, double x0) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double q[3][3];
  double x, z;
  double n[3];
  double theta;
  double q0;

  assert(fq);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  q0 = param->q0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    x = noffset[X] + ic;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	z = noffset[Z] + kc;
	
	index = coords_index(ic, jc, kc);

	theta = 1.0*N/2.0*atan2((1.0*z-z0),(1.0*x-x0)) + q0*(z-z0);	
	n[X] = cos(theta);
	n[Y] = sin(theta);
	n[Z] = 0.0;

	fe_lc_q_uniaxial(param, n, q);
	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_random_q_init
 *
 *  Set a decomposition-independent random initial Q tensor
 *  based on the initial order amplitude0 and a randomly
 *  chosen director at each site.
 *
 *****************************************************************************/

int blue_phase_random_q_init(fe_lc_param_t * param, field_t * fq) {

  int ic, jc, kc, index;
  int nlocal[3];
  int seed = DEFAULT_SEED;

  double n[3];            /* random director */
  double q[3][3];         /* resulting unixial q */
  double phase1, phase2;

  double ran1, ran2;
  pe_t * pe = NULL;
  cs_t * cs = NULL;
  noise_t * rng = NULL;
  PI_DOUBLE(pi);

  assert(fq);
  
  pe_ref(&pe);
  cs_ref(&cs);
  coords_nlocal(nlocal);

  noise_create(pe, cs, &rng);
  noise_init(rng, seed);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	    
	index = coords_index(ic, jc, kc);

	noise_uniform_double_reap(rng, index, &ran1);
	noise_uniform_double_reap(rng, index, &ran2);

	phase1 = 2.0*pi*(0.5 - ran1);
	phase2 = acos(2.0*ran2 - 1.0);

	n[X] = cos(phase1)*sin(phase2);
	n[Y] = sin(phase1)*sin(phase2);
	n[Z] = cos(phase2);

	fe_lc_q_uniaxial(param, n, q);
	field_tensor_set(fq, index, q);
      }
    }
  }

  noise_free(rng);

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_random_q_rectangle
 *
 *  Within the limits of the rectanular box provided, set the initial
 *  Q tensor to a random value. The idea here is to 'melt' the order
 *  set previously (e.g., to cholesteric), but only in a small region.
 *
 *  We should then have amplitude of order a0 << amplitude0; we use
 *  a0 = 1.0e-06. 
 * 
 *****************************************************************************/

int blue_phase_random_q_rectangle(fe_lc_param_t * param, field_t * fq, int rmin[3],
				  int rmax[3]) {

  int ic, jc, kc, index;
  int ia, ib;
  int nlocal[3];
  int noffset[3];
  int seed = DEFAULT_SEED;

  double n[3];
  double q[3][3];
  double phase1, phase2;
  double a0 = 0.01;             /* Initial amplitude of order in 'box' */

  double ran1, ran2;

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  noise_t * rng = NULL;
  PI_DOUBLE(pi);
  KRONECKER_DELTA_CHAR(d);

  assert(fq);

  pe_ref(&pe);
  cs_ref(&cs);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  noise_create(pe, cs, &rng);
  noise_init(rng, seed);

  /* Adjust min, max to allow for parallel offset of box */

  rmin[X] -= noffset[X]; rmax[X] -= noffset[X];
  rmin[Y] -= noffset[Y]; rmax[Y] -= noffset[Y];
  rmin[Z] -= noffset[Z]; rmax[Z] -= noffset[Z];

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	if (ic < rmin[X] || ic > rmax[X] || jc < rmin[Y] || jc > rmax[Y] ||
	    kc < rmin[Z] || kc > rmax[Z]) continue;

	index = coords_index(ic, jc, kc);

	noise_uniform_double_reap(rng, index, &ran1);
	noise_uniform_double_reap(rng, index, &ran2);

	phase1 = 2.0*pi*(0.5 - ran1);
	phase2 = acos(2.0*ran2 - 1.0);
	    
	n[X] = cos(phase1)*sin(phase2);
	n[Y] = sin(phase1)*sin(phase2);
	n[Z] = cos(phase2);

	/* Uniaxial approximation using a0 */
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    q[ia][ib] = 0.5*a0*(3.0*n[ia]*n[ib] - d[ia][ib]);
	  }
	}
	field_tensor_set(fq, index, q);

      }
    }
  }

  noise_free(rng);

  return 0;
}


/****************************************************************************
 *
 *  M_rot
 *
 *  Matrix for rotation around specified axis
 *
 ****************************************************************************/

void blue_phase_M_rot(double M[3][3], int dim, double alpha){

  if(dim==0){
    M[0][0] = 1.0;
    M[0][1] = 0.0;
    M[0][2] = 0.0;
    M[1][0] = 0.0;
    M[1][1] = cos(alpha);
    M[1][2] = -sin(alpha);
    M[2][0] = 0.0;
    M[2][1] = sin(alpha);
    M[2][2] = cos(alpha);
  }

  if(dim==1){
    M[0][0] = cos(alpha);
    M[0][1] = 0.0;
    M[0][2] = -sin(alpha);
    M[1][0] = 0.0;
    M[1][1] = 1.0;
    M[1][2] = 0.0;
    M[2][0] = sin(alpha);
    M[2][1] = 0.0;
    M[2][2] = cos(alpha);
  }

  if(dim==2){
    M[0][0] = cos(alpha);
    M[0][1] = -sin(alpha);
    M[0][2] = 0.0;
    M[1][0] = sin(alpha);
    M[1][1] = cos(alpha);
    M[1][2] = 0.0;
    M[2][0] = 0.0;
    M[2][1] = 0.0;
    M[2][2] = 1.0;
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_cf1_init
 *
 *  Initialise a cholesteric finger of the first kind.
 *  Uses the current free energy parameters
 *     q0 (P=2pi/q0)
 *
 *****************************************************************************/

int blue_phase_cf1_init(fe_lc_param_t * param, field_t * fq, int axis) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int ntotal[3];
  int index;

  double n[3];
  double q[3][3];
  double q0;
  double alpha, alpha0, beta;
  PI_DOUBLE(pi);

  assert(fq);
  assert(axis == X || axis == Y || axis == Z);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  coords_ntotal(ntotal);

  q0 = param->q0;
  alpha0 = 0.5*pi; 

  n[X] = 0.0;
  n[Y] = 0.0;
  n[Z] = 0.0;
 
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	if (axis == X) {

	    alpha = alpha0*sin(pi*(noffset[Z]+kc)/ntotal[Z]);
	    beta  = -2.0*(pi*(noffset[Z]+kc)/ntotal[Z]-0.5*pi);

	    n[X]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[Y]+jc)) -
		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
		     sin(alpha)*sin(beta)*cos(alpha);
	    n[Y]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[Y]+jc)) -
		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
		     sin(alpha)*cos(beta)*cos(alpha);
	    n[Z]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
		     cos(alpha)*cos(alpha);

	}

	if (axis == Y) {

	    alpha = alpha0*sin(pi*(noffset[X]+ic)/ntotal[X]);
	    beta  = -2.0*(pi*(noffset[X]+ic)/ntotal[X]-0.5*pi);

	    n[Y]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[Z]+kc)) -
		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
		     sin(alpha)*sin(beta)*cos(alpha);
	    n[Z]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[Z]+kc)) -
		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
		     sin(alpha)*cos(beta)*cos(alpha);
	    n[X]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
		     cos(alpha)*cos(alpha);

	}

	if (axis == Z) {

	    alpha = alpha0*sin(pi*(noffset[Y]+jc)/ntotal[Y]);
	    beta  = -2.0*(pi*(noffset[Y]+jc)/ntotal[Y]-0.5*pi);

	    n[Z]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[X]+ic)) -
		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
		     sin(alpha)*sin(beta)*cos(alpha);
	    n[X]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[X]+ic)) -
		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
		     sin(alpha)*cos(beta)*cos(alpha);
	    n[Y]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
		     cos(alpha)*cos(alpha);

	}

	fe_lc_q_uniaxial(param, n, q);
	field_tensor_set(fq, index, q);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_random_cf1_init
 *
 *  Initialise a cholesteric finger of the first kind.
 *  In addtion to blue_phase_cf1_init a traceless symmetric
 *  random fluctuation is put on the Q-tensor.
 *
 *  Uses the current free energy parameters
 *     q0 (pitch = 2pi/q0)
 *
 *****************************************************************************/

int blue_phase_random_cf1_init(fe_lc_param_t * param, field_t * fq, int axis) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int ntotal[3];
  int index;

  double n[3];
  double q[3][3];
  double q0;
  double alpha, alpha0, beta;

  int ia, ib, id;
  double tmatrix[3][3][NQAB];
  double chi[NQAB], chi_qab[3][3];
  double var = 1e-1; /* variance of random fluctuation */
  int seed = DEFAULT_SEED;

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  noise_t * rng = NULL;
  PI_DOUBLE(pi);

  assert(fq);
  assert(axis == X || axis == Y || axis == Z);

  pe_ref(&pe);
  cs_ref(&cs);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  coords_ntotal(ntotal);

  q0 = param->q0;
  alpha0 = 0.5*pi; 

  n[X] = 0.0;
  n[Y] = 0.0;
  n[Z] = 0.0;

  noise_create(pe, cs, &rng);
  noise_init(rng, seed);
 
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	if (axis == X) {

	    alpha = alpha0*sin(pi*(noffset[Z]+kc)/ntotal[Z]);
	    beta  = -2.0*(pi*(noffset[Z]+kc)/ntotal[Z]-0.5*pi);

	    n[X]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[Y]+jc)) -
		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
		     sin(alpha)*sin(beta)*cos(alpha);
	    n[Y]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[Y]+jc)) -
		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
		     sin(alpha)*cos(beta)*cos(alpha);
	    n[Z]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
		     cos(alpha)*cos(alpha);

	}

	if (axis == Y) {

	    alpha = alpha0*sin(pi*(noffset[X]+ic)/ntotal[X]);
	    beta  = -2.0*(pi*(noffset[X]+ic)/ntotal[X]-0.5*pi);

	    n[Y]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[Z]+kc)) -
		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
		     sin(alpha)*sin(beta)*cos(alpha);
	    n[Z]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[Z]+kc)) -
		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
		     sin(alpha)*cos(beta)*cos(alpha);
	    n[X]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
		     cos(alpha)*cos(alpha);

	}

	if (axis == Z) {

	    alpha = alpha0*sin(pi*(noffset[Y]+jc)/ntotal[Y]);
	    beta  = -2.0*(pi*(noffset[Y]+jc)/ntotal[Y]-0.5*pi);

	    n[Z]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[X]+ic)) -
		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
		     sin(alpha)*sin(beta)*cos(alpha);
	    n[X]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[X]+ic)) -
		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
		     sin(alpha)*cos(beta)*cos(alpha);
	    n[Y]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
		     cos(alpha)*cos(alpha);

	}

	fe_lc_q_uniaxial(param, n, q);

	/* Random fluctuation with specified variance */
        noise_reap_n(rng, index, NQAB, chi);

	for (id = 0; id < NQAB; id++) {
	  chi[id] = var*chi[id];
	}

	beris_edw_tmatrix(tmatrix);

	/* Random fluctuation added to tensor order parameter */
	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    chi_qab[ia][ib] = 0.0;
	    for (id = 0; id < NQAB; id++) {
	      chi_qab[ia][ib] += chi[id]*tmatrix[ia][ib][id];
	    }
	    q[ia][ib] += chi_qab[ia][ib];
	  }
	}

	field_tensor_set(fq, index, q);

      }
    }
  }

  noise_free(rng);

  return 0;
}
