/*****************************************************************************
 *
 *  blue_phase.c
 *
 *  Routines related to blue phase liquid crystal free energy
 *  and molecular field.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "field.h"
#include "field_s.h"
#include "field_grad.h"
#include "field_grad_s.h"
#include "blue_phase.h"
#include "io_harness.h"
#include "leesedwards.h"
#include "physics.h"
#include "control.h"
#include "colloids_Q_tensor.h"

static double q0_;        /* Pitch = 2pi / q0_ */
static double a0_;        /* Bulk free energy parameter A_0 */
static double gamma_;     /* Controls magnitude of order */
static double kappa0_;    /* Elastic constant \kappa_0 */
static double kappa1_;    /* Elastic constant \kappa_1 */

static double xi_;        /* effective molecular aspect ratio (<= 1.0) */
static double redshift_;  /* redshift parameter */
static double rredshift_; /* reciprocal */
static double zeta_;      /* Apolar activity parameter \zeta */

static int redshift_update_ = 0; /* Dynamic cubic redshift update */
static double epsilon_ = 0.0;    /* Dielectric anisotropy (e/12pi) */

static const double redshift_min_ = 0.00000000001; 

static field_t * q_ = NULL;
static field_grad_t * grad_q_ = NULL;

static io_info_t * io_info_fed;

static int fed_write(FILE *, int index, void * self);
static int fed_write_ascii(FILE *, int index, void * self);

/* structure containing constants used in blue phase kernels */ 
bluePhaseKernelConstants_t bpc; //host copy
__targetConst__ bluePhaseKernelConstants_t tbpc; //target copy


/*****************************************************************************
 *
 *  blue_phase_set_kernel_constants
 *
 *  initialise the components of the blue phase kernel constant structure
 *  on both the host and the target
 * 
 *****************************************************************************/

__targetHost__ void blue_phase_set_kernel_constants(){


  int nlocal[3];
  int nextra = 1;
  
  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  double redshift_ = blue_phase_redshift(); 
  double rredshift_ = blue_phase_rredshift(); 

  bpc.q0=blue_phase_q0();
  bpc.q0 = bpc.q0*rredshift_;

  bpc.a0_=blue_phase_a0();
  bpc.kappa0=blue_phase_kappa0();
  bpc.kappa1=blue_phase_kappa1();
  bpc.kappa0 = bpc.kappa0*redshift_*redshift_;
  bpc.kappa1 = bpc.kappa1*redshift_*redshift_;

  bpc.xi_=blue_phase_get_xi();
  bpc.zeta_=blue_phase_get_zeta();
  bpc.gamma_=blue_phase_gamma();
  bpc.epsilon_=blue_phase_dielectric_anisotropy();

  double e0_frequency;
  double coswt;
  int t_step;
  int ia;

  physics_e0(bpc.e0);
  physics_e0_frequency(&e0_frequency);
  
  /* Scale amplitude if e0_frequency != 0 */ 
  if (e0_frequency) {
    t_step = get_step();
    coswt  = cos(2.0*pi_*e0_frequency*t_step);
    for (ia = 0; ia < 3; ia++) {
      bpc.e0[ia] *= coswt;
    }
  }
  
  bpc.nhalo = coords_nhalo();
  bpc.Nall[X]=nlocal[X]+2*bpc.nhalo;  bpc.Nall[Y]=nlocal[Y]+2*bpc.nhalo;  bpc.Nall[Z]=nlocal[Z]+2*bpc.nhalo;
  bpc.nSites=bpc.Nall[X]*bpc.Nall[Y]*bpc.Nall[Z];
  bpc.nextra=nextra;
  memcpy(bpc.d_,d_,3*3*sizeof(double));
  memcpy(bpc.e_,e_,3*3*3*sizeof(double));
  bpc.r3_=r3_;

  blue_phase_coll_w12(&(bpc.w1_coll), &(bpc.w2_coll));
  blue_phase_wall_w12(&(bpc.w1_wall), &(bpc.w2_wall));
  
  bpc.amplitude = blue_phase_amplitude_compute(); 

  copyConstToTarget(&tbpc, &bpc, sizeof(bluePhaseKernelConstants_t)); 

  return;
}


/*****************************************************************************
 *
 *  blue_phase_host_constant_ptr
 *
 *  get a pointer to the host version of the kernel constant structure
 *
 *****************************************************************************/

__targetHost__ void blue_phase_host_constant_ptr(void** tmp){

  *tmp=&bpc;

}



/*****************************************************************************
 *
 *  blue_phase_target_constant_ptr
 *
 *  get a pointer to the target version of the kernel constant structure
 *
 *****************************************************************************/

__targetHost__ void blue_phase_target_constant_ptr(void** tmp){


  targetConstAddress(tmp,tbpc);
  return;

}


/*****************************************************************************
 *
 *  blue_phase_q_set
 *
 *  Attach a reference to the order parameter field object, and the
 *  associated gradient object.
 *
 *****************************************************************************/

__targetHost__ int blue_phase_q_set(field_t * q, field_grad_t * dq) {

  assert(q);
  assert(dq);

  q_ = q;
  grad_q_ = dq;

  return 0;
}

/*****************************************************************************
 *
 *  blue_phase_set_free_energy_parameters
 *
 *  Enforces the 'one constant approximation' kappa0 = kappa1 = kappa
 *
 *  Note that these values remain unchanged throughout. Redshifted
 *  values are computed separately as needed.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_set_free_energy_parameters(double a0, double gamma,
					   double kappa, double q0) {
  a0_ = a0;
  gamma_ = gamma;
  kappa0_ = kappa;
  kappa1_ = kappa;
  q0_ = q0;

  /* Anchoring boundary conditions require kappa0 from free energy */
  fe_kappa_set(kappa0_);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_amplitude_compute
 *
 *  Scalar order parameter in the nematic state, minimum of bulk free energy 
 *
 *****************************************************************************/

__targetHost__ double blue_phase_amplitude_compute(void) {

  double amplitude;
  
  amplitude = 2.0/3.0*(0.25 + 0.75*sqrt(1.0 - 8.0/(3.0*gamma_)));

  return amplitude;
}

/*****************************************************************************
 *
 *  blue_phase_set_xi
 *
 *  Set the molecular aspect ratio.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_set_xi(double xi) {

  xi_ = xi;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_get_xi
 *
 *****************************************************************************/

__targetHost__ double blue_phase_get_xi(void) {

  return xi_;
}

/*****************************************************************************
 *
 *  blue_phase_set_zeta
 *
 *  Set the activity parameter.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_set_zeta(double zeta) {

  zeta_ = zeta;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_get_zeta
 *
 *****************************************************************************/

__targetHost__ double blue_phase_get_zeta(void) {

  return zeta_;
}

/*****************************************************************************
 *
 *  blue_phase_set_gamma
 *
 *  Set the gamma_ parameter.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_set_gamma(double gamma) {

  gamma_ = gamma;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_get_gamma
 *
 *****************************************************************************/

__targetHost__ double blue_phase_get_gamma(void) {

  return gamma_;
}

/*****************************************************************************
 *
 *  blue_phase_free_energy_density
 *
 *  Return the free energy density at lattice site index.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_free_energy_density(const int index) {



  double e;
  double q[3][3];
  double dq[3][3][3];

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);

  //we are doing this on the host
  blue_phase_set_kernel_constants();
  void* pcon=NULL;
  blue_phase_host_constant_ptr(&pcon);

  e = blue_phase_compute_fed(q, dq, (bluePhaseKernelConstants_t*) pcon);
  return e;

}

/*****************************************************************************
 *
 *  blue_phase_compute_fed
 *
 *  Compute the free energy density as a function of q and the q gradient
 *  tensor dq.
 *
 *****************************************************************************/

__targetHost__ __target__ double blue_phase_compute_fed(double q[3][3], 
							double dq[3][3][3],
					 bluePhaseKernelConstants_t* pbpc) {

  int ia, ib, ic, id;
  double q2, q3;
  double dq0, dq1;
  double sum;
  double efield;

  q2 = 0.0;

  /* Q_ab^2 */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += q[ia][ib]*q[ia][ib];
    }
  }

  /* Q_ab Q_bc Q_ca */

  q3 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	/* We use here the fact that q[ic][ia] = q[ia][ic] */
	q3 += q[ia][ib]*q[ib][ic]*q[ia][ic];
      }
    }
  }

  /* (d_b Q_ab)^2 */

  dq0 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    sum = 0.0;
    for (ib = 0; ib < 3; ib++) {
      sum += dq[ib][ia][ib];
    }
    dq0 += sum*sum;
  }

  /* (e_acd d_c Q_db + 2q_0 Q_ab)^2 */
  /* With symmetric Q_db write Q_bd */

  dq1 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum += pbpc->e_[ia][ic][id]*dq[ic][ib][id];
	}
      }
      sum += 2.0*pbpc->q0*q[ia][ib];
      dq1 += sum*sum;
    }
  }

  /* Electric field term (epsilon_ includes the factor 1/12pi) */

  efield = 0.0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      efield += pbpc->e0[ia]*q[ia][ib]*pbpc->e0[ib];
    }
  }

  sum = 0.5*pbpc->a0_*(1.0 - pbpc->r3_*pbpc->gamma_)*q2 - pbpc->r3_*pbpc->a0_*pbpc->gamma_*q3 +
    0.25*pbpc->a0_*pbpc->gamma_*q2*q2 + 0.5*pbpc->kappa0*dq0 + 0.5*pbpc->kappa1*dq1 - pbpc->epsilon_*efield;


  return sum;
}

/*****************************************************************************
 *
 *  blue_phase_compute_bulk_fed
 *
 *  Compute the bulk free energy density as a function of q.
 *
 *  Note: This function contains also the part quadratic in q 
 *        which is normally part of the gradient free energy. 
 *
 *****************************************************************************/

__targetHost__ double blue_phase_compute_bulk_fed(double q[3][3]) {

  int ia, ib, ic;
  double q0;
  double kappa1;
  double q2, q3;
  double sum;

  q0 = q0_*rredshift_;
  kappa1 = kappa1_*redshift_*redshift_;

  q2 = 0.0;

  /* Q_ab^2 */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += q[ia][ib]*q[ia][ib];
    }
  }

  /* Q_ab Q_bc Q_ca */

  q3 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
	/* We use here the fact that q[ic][ia] = q[ia][ic] */
	q3 += q[ia][ib]*q[ib][ic]*q[ia][ic];
      }
    }
  }

  sum = 0.5*a0_*(1.0 - r3_*gamma_)*q2 - r3_*a0_*gamma_*q3 + 0.25*a0_*gamma_*q2*q2;

  /* Add terms quadratic in q from gradient free energy */ 
  sum += 0.5*kappa1*4.0*q0*q0*q2;

  return sum;
}

/*****************************************************************************
 *
 *  blue_phase_compute_gradient_fed
 *
 *  Compute the gradient contribution to the free energy density 
 *  as a function of q and the q gradient tensor dq.
 *
 *  Note: The part quadratic in q has been added to the bulk free energy.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_compute_gradient_fed(double q[3][3], 
						      double dq[3][3][3]) {

  int ia, ib, ic, id;
  double q0;
  double kappa0, kappa1;
  double dq0, dq1;
  double q2;
  double sum;

  q0 = q0_*rredshift_;
  kappa0 = kappa0_*redshift_*redshift_;
  kappa1 = kappa1_*redshift_*redshift_;

  /* (d_b Q_ab)^2 */

  dq0 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    sum = 0.0;
    for (ib = 0; ib < 3; ib++) {
      sum += dq[ib][ia][ib];
    }
    dq0 += sum*sum;
  }

  /* (e_acd d_c Q_db + 2q_0 Q_ab)^2 */
  /* With symmetric Q_db write Q_bd */

  dq1 = 0.0;
  q2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      sum = 0.0;
  
      q2 += q[ia][ib]*q[ia][ib];

      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum += e_[ia][ic][id]*dq[ic][ib][id];
	}
      }
      sum += 2.0*q0*q[ia][ib];
      dq1 += sum*sum;
    }
  }

  /* Subtract part that is quadratic in q */
  dq1 -= 4.0*q0*q0*q2;

  sum = 0.5*kappa0*dq0 + 0.5*kappa1*dq1;

  return sum;
}

/*****************************************************************************
 *
 *  blue_phase_molecular_field
 *
 *  Return the molcular field h[3][3] at lattice site index.
 *
 *  Note this is only valid in the one-constant approximation at
 *  the moment (kappa0 = kappa1 = kappa).
 *
 *****************************************************************************/

__targetHost__ void blue_phase_molecular_field(int index, double h[3][3]) {


  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  assert(kappa0_ == kappa1_);

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);
  field_grad_tensor_delsq(grad_q_, index, dsq);

  //we are doing this on the host
  blue_phase_set_kernel_constants();
  void* pcon=NULL;
  blue_phase_host_constant_ptr(&pcon);
 
  blue_phase_compute_h(q, dq, dsq, h, (bluePhaseKernelConstants_t*) pcon);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_compute_h
 *
 *  Compute the molcular field h from q, the q gradient tensor dq, and
 *  the del^2 q tensor.
 *
 *****************************************************************************/

__targetHost__ __target__ void blue_phase_compute_h(double q[3][3], 
						    double dq[3][3][3],
						    double dsq[3][3], 
						    double h[3][3],
						    bluePhaseKernelConstants_t* pbpc) {
  int ia, ib, ic, id;

  double q2;
  double e2;
  double eq;
  double sum;

  /* From the bulk terms in the free energy... */

  q2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += q[ia][ib]*q[ia][ib];
    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	sum += q[ia][ic]*q[ib][ic];
      }
      h[ia][ib] = -pbpc->a0_*(1.0 - pbpc->r3_*pbpc->gamma_)*q[ia][ib]
	+ pbpc->a0_*pbpc->gamma_*(sum - pbpc->r3_*q2*pbpc->d_[ia][ib]) - pbpc->a0_*pbpc->gamma_*q2*q[ia][ib];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  eq = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
	eq += pbpc->e_[ib][ic][ia]*dq[ib][ic][ia];
      }
    }
  }

  /* d_c Q_db written as d_c Q_bd etc */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  sum +=
	    (pbpc->e_[ia][ic][id]*dq[ic][ib][id] + pbpc->e_[ib][ic][id]*dq[ic][ia][id]);
	}
      }
      h[ia][ib] += pbpc->kappa0*dsq[ia][ib]
	- 2.0*pbpc->kappa1*pbpc->q0*sum + 4.0*pbpc->r3_*pbpc->kappa1*pbpc->q0*eq*pbpc->d_[ia][ib]
	- 4.0*pbpc->kappa1*pbpc->q0*pbpc->q0*q[ia][ib];
    }
  }

  /* Electric field term */

  e2 = 0.0;
  for (ia = 0; ia < 3; ia++) {
    e2 += pbpc->e0[ia]*pbpc->e0[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] +=  pbpc->epsilon_*(pbpc->e0[ia]*pbpc->e0[ib] - pbpc->r3_*pbpc->d_[ia][ib]*e2);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_chemical_stress
 *
 *  Return the stress sth[3][3] at lattice site index.
 *
 *****************************************************************************/


__targetHost__ void blue_phase_chemical_stress(int index, double sth[3][3]) {

  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);
  field_grad_tensor_delsq(grad_q_, index, dsq);

  //we are doing this on the host
  blue_phase_set_kernel_constants();
  void* pcon=NULL;
  blue_phase_host_constant_ptr(&pcon);


  blue_phase_compute_h(q, dq, dsq, h, (bluePhaseKernelConstants_t*) pcon);
  blue_phase_compute_stress(q, dq, h, sth, (bluePhaseKernelConstants_t*) pcon);

  return;
}




//targetDP development version
__targetHost__ __target__ void blue_phase_chemical_stress_dev(int index, field_t* t_q, field_grad_t* t_q_grad, double* t_pth, void* pcon, int calledFromPhiForceStress) { 


  if (calledFromPhiForceStress!=1){
  #ifndef CUDA
  fatal("Error: in porting to targetDP we are assuming that blue_phase_chemical_stress is only called from phi_force_stress\n");
  #endif  
  }
 
  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  double sth_loc[3][3];

  int ia, ib;

  bluePhaseKernelConstants_t* pbpc= (bluePhaseKernelConstants_t*) pcon;

  q[X][X] = t_q->data[NQAB*index + XX];
  q[X][Y] = t_q->data[NQAB*index + XY];
  q[X][Z] = t_q->data[NQAB*index + XZ];
  q[Y][X] = q[X][Y];
  q[Y][Y] = t_q->data[NQAB*index + YY];
  q[Y][Z] = t_q->data[NQAB*index + YZ];
  q[Z][X] = q[X][Z];
  q[Z][Y] = q[Y][Z];
  q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];

  for (ia = 0; ia < NVECTOR; ia++) {
    dq[ia][X][X] = t_q_grad->grad[NVECTOR*(NQAB*index + XX) + ia];
    dq[ia][X][Y] = t_q_grad->grad[NVECTOR*(NQAB*index + XY) + ia];
    dq[ia][X][Z] = t_q_grad->grad[NVECTOR*(NQAB*index + XZ) + ia];
    dq[ia][Y][X] = dq[ia][X][Y];
    dq[ia][Y][Y] = t_q_grad->grad[NVECTOR*(NQAB*index + YY) + ia];
    dq[ia][Y][Z] = t_q_grad->grad[NVECTOR*(NQAB*index + YZ) + ia];
    dq[ia][Z][X] = dq[ia][X][Z];
    dq[ia][Z][Y] = dq[ia][Y][Z];
    dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
  }


  dsq[X][X] = t_q_grad->delsq[NQAB*index + XX];
  dsq[X][Y] = t_q_grad->delsq[NQAB*index + XY];
  dsq[X][Z] = t_q_grad->delsq[NQAB*index + XZ];
  dsq[Y][X] = dsq[X][Y];
  dsq[Y][Y] = t_q_grad->delsq[NQAB*index + YY];
  dsq[Y][Z] = t_q_grad->delsq[NQAB*index + YZ];
  dsq[Z][X] = dsq[X][Z];
  dsq[Z][Y] = dsq[Y][Z];
  dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];


  blue_phase_compute_h(q, dq, dsq, h, pbpc);
  blue_phase_compute_stress(q, dq, h, sth_loc, pbpc);


  for(ia=0;ia<3;ia++)
    for(ib=0;ib<3;ib++)
      t_pth[index*9+ia*3+ib]=sth_loc[ia][ib];
  

  return;
}

/*****************************************************************************
 *
 *  blue_phase_compute_stress
 *
 *  Compute the stress as a function of the q tensor, the q tensor
 *  gradient and the molecular field.
 *
 *  Note the definition here has a minus sign included to allow
 *  computation of the force as minus the divergence (which often
 *  appears as plus in the liquid crystal literature). This is a
 *  separate operation at the end to avoid confusion.
 *
 *****************************************************************************/

__targetHost__ __target__ void blue_phase_compute_stress(double q[3][3], double dq[3][3][3],
					  double h[3][3], double sth[3][3], 
					  bluePhaseKernelConstants_t* pbpc) {
  int ia, ib, ic, id, ie;

  double qh;
  double p0;

  /* We have ignored the rho T term at the moment, assumed to be zero
     (in particular, it has no divergence if rho = const). */

  p0 = 0.0 - blue_phase_compute_fed(q, dq, pbpc);

  /* The contraction Q_ab H_ab */

  qh = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      qh += q[ia][ib]*h[ia][ib];
    }
  }

  /* The term in the isotropic pressure, plus that in qh */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] = -p0*pbpc->d_[ia][ib] + 2.0*pbpc->xi_*(q[ia][ib]
  						 + pbpc->r3_*pbpc->d_[ia][ib])*qh;
    }
  }

  /* Remaining two terms in xi and molecular field */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
  	sth[ia][ib] +=
  	  -pbpc->xi_*h[ia][ic]*(q[ib][ic] + pbpc->r3_*pbpc->d_[ib][ic])
  	  -pbpc->xi_*(q[ia][ic] + pbpc->r3_*pbpc->d_[ia][ic])*h[ib][ic];
      }
    }
  }

  /* Dot product term d_a Q_cd . dF/dQ_cd,b */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      for (ic = 0; ic < 3; ic++) {
  	for (id = 0; id < 3; id++) {
  	  sth[ia][ib] +=
  	    - pbpc->kappa0*dq[ia][ib][ic]*dq[id][ic][id]
  	    - pbpc->kappa1*dq[ia][ic][id]*dq[ib][ic][id]
  	    + pbpc->kappa1*dq[ia][ic][id]*dq[ic][ib][id];

  	  for (ie = 0; ie < 3; ie++) {
  	    sth[ia][ib] +=
  	      -2.0*pbpc->kappa1*pbpc->q0*dq[ia][ic][id]*pbpc->e_[ib][ic][ie]*q[id][ie];
  	  }
  	}
      }
    }
  }

  /* The antisymmetric piece q_ac h_cb - h_ac q_cb. We can
   * rewrite it as q_ac h_bc - h_ac q_bc. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for (ic = 0; ic < 3; ic++) {
  	sth[ia][ib] += q[ia][ic]*h[ib][ic] - h[ia][ic]*q[ib][ic];
      }
    }
  }

  /* Additional active stress -zeta*(q_ab - 1/3 d_ab) */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] -= pbpc->zeta_*(q[ia][ib] + pbpc->r3_*pbpc->d_[ia][ib]);
    }
  }

  /* This is the minus sign. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
  	sth[ia][ib] = -sth[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_q_uniaxial
 *
 *  For given director n we return
 *
 *     Q_ab = (1/2) A (3 n_a n_b - d_ab)
 *
 *  where A gives the maximum amplitude of order on diagonalisation.
 *
 *  Note this is slightly different  from the definition in
 *  Wright and Mermin (Eq. 4.3) where
 *
 *     Q_ab = (1/3) gamma (3 n_a n_b - d_ab)
 *
 *  and the magnitude of order is then (2/3) gamma.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_q_uniaxial(double amp, const double n[3], double q[3][3]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q[ia][ib] = 0.5*amp*(3.0*n[ia]*n[ib] - d_[ia][ib]);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  blue_phase_chirality
 *
 *  Return the chirality, which is defined here as
 *         sqrt(108 \kappa_0 q_0^2 / A_0 \gamma)
 *
 *  Not dependent on the redshift.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_chirality(void) {

  double chirality;

  chirality = sqrt(108.0*kappa0_*q0_*q0_ / (a0_*gamma_));

  return chirality;
}

/*****************************************************************************
 *
 *  blue_phase_reduced_temperature
 *
 *  Return the the reduced temperature defined here as
 *       27*(1 - \gamma/3) / \gamma
 *
 *****************************************************************************/

__targetHost__ double blue_phase_reduced_temperature(void) {

  double tau;

  tau = 27.0*(1.0 - r3_*gamma_) / gamma_;

  return tau;
}

/*****************************************************************************
 *
 *  blue_phase_dimensionless_field_strength
 *
 *  Return the dimensionless field strength which is
 *      e^2 = (27 epsilon / 32 pi A_O gamma) E_a E_a
 *
 *****************************************************************************/

__targetHost__ double blue_phase_dimensionless_field_strength(void) {

  int ia;
  double e;
  double fieldsq;
  double e0[3];

  physics_e0(e0);

  fieldsq = 0.0;
  for (ia = 0; ia < 3; ia++) {
    fieldsq += e0[ia]*e0[ia];
  }

  /* Remember epsilon is stored with factor (1/12pi) */ 

  e = sqrt(27.0*(12.0*pi_*epsilon_)*fieldsq/(32.0*pi_*a0_*gamma_));

  return e;
}

/*****************************************************************************
 *
 *  blue_phase_redshift
 *
 *  Return the redshift parameter.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_redshift(void) {

  return redshift_;
}

/*****************************************************************************
 *
 *  blue_phase_rredshift
 *
 *  Return the reciprocal redshift parameter.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_rredshift(void) {

  return rredshift_;
}
/*****************************************************************************
 *
 *  blue_phase_kappa0
 *
 *  Return the first elastic constant.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_kappa0(void) {

  return kappa0_;
}

/*****************************************************************************
 *
 *  blue_phase_kappa1
 *
 *  Return the second elastic constant.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_kappa1(void) {

  return kappa1_;
}

/*****************************************************************************
 *
 *  blue_phase_a0
 *
 *  Return the bulk free energy constant.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_a0(void) {

  return a0_;
}

/*****************************************************************************
 *
 *  blue_phase_gamma
 *
 *  Return the inversed effective temperature.
 *
 *****************************************************************************/

__targetHost__ double blue_phase_gamma(void) {

  return gamma_;
}

/*****************************************************************************
 *
 *  blue_phase_q0
 *
 *  Return the pitch wavenumber (unredshifted).
 *
 *****************************************************************************/

__targetHost__ double blue_phase_q0(void) {

  return q0_;
}

/*****************************************************************************
 *
 *  blue_phase_redshift_set
 *
 *  Set the redshift parameter.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_redshift_set(const double redshift) {

  assert(fabs(redshift) >= redshift_min_);
  redshift_ = redshift;
  rredshift_ = 1.0/redshift_;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_redshift_update_set
 *
 *  At the moment the 'token' is on/off.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_redshift_update_set(int update) {

  redshift_update_ = update;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_redshift_compute
 *
 *  Redshift adjustment. If this is required at all, it should be
 *  done at every timestep. It gives rise to an Allreduce.
 *
 *  The redshift calculation uses the unredshifted values of the
 *  free energy parameters kappa0_, kappa1_ and q0_.
 *
 *  The term quadratic in gradients may be written F_ddQ
 *
 *     (1/2) [ kappa1 (d_a Q_bc)^2 - kappa1 (d_a Q_bc d_b Q_ac)
 *           + kappa0 (d_b Q_ab)^2 ]
 *
 *  The linear term is F_dQ
 *
 *     2 q0 kappa1 Q_ab e_acg d_c Q_gb
 *
 *  The new redshift is computed as - F_dQ / 2 F_ddQ
 *
 *****************************************************************************/

__targetHost__ void blue_phase_redshift_compute(void) {

  int ic, jc, kc, index;
  int ia, ib, id, ig;
  int nlocal[3];

  double q[3][3], dq[3][3][3];

  double dq0, dq1, dq2, dq3, sum;
  double egrad_local[2], egrad[2];    /* Gradient terms for redshift calc. */
  double rnew;

  if (redshift_update_ == 0) return;

  coords_nlocal(nlocal);

  egrad_local[0] = 0.0;
  egrad_local[1] = 0.0;

  /* Accumulate the sums (all fluid) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	field_tensor(q_, index, q);
	field_grad_tensor_grad(grad_q_, index, dq);

	/* kappa0 (d_b Q_ab)^2 */

	dq0 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  sum = 0.0;
	  for (ib = 0; ib < 3; ib++) {
	    sum += dq[ib][ia][ib];
	  }
	  dq0 += sum*sum;
	}

	/* kappa1 (e_agd d_g Q_db + 2q_0 Q_ab)^2 */

	dq1 = 0.0;
	dq2 = 0.0;
	dq3 = 0.0;

	for (ia = 0; ia < 3; ia++) {
	  for (ib = 0; ib < 3; ib++) {
	    sum = 0.0;
	    for (ig = 0; ig < 3; ig++) {
	      dq1 += dq[ia][ib][ig]*dq[ia][ib][ig];
	      dq2 += dq[ia][ib][ig]*dq[ib][ia][ig];
	      for (id = 0; id < 3; id++) {
		sum += e_[ia][ig][id]*dq[ig][id][ib];
	      }
	    }
	    dq3 += q[ia][ib]*sum;
	  }
	}

	/* linear gradient and square gradient terms */

	egrad_local[0] += 2.0*q0_*kappa1_*dq3;
	egrad_local[1] += 0.5*(kappa1_*dq1 - kappa1_*dq2 + kappa0_*dq0);

      }
    }
  }

  /* Allreduce the gradient results, and compute a new redshift (we
   * keep the old one if problematic). */

  MPI_Allreduce(egrad_local, egrad, 2, MPI_DOUBLE, MPI_SUM, cart_comm());

  rnew = redshift_;
  if (egrad[1] != 0.0) rnew = -0.5*egrad[0]/egrad[1];
  if (fabs(rnew) < redshift_min_) rnew = redshift_;

  blue_phase_redshift_set(rnew);

  return;
}

/*****************************************************************************
 *
 *  blue_phase_dielectric_anisotropy_set
 *
 *  Include the factor 1/12pi appearing in the free energy.
 *
 *****************************************************************************/

__targetHost__ void blue_phase_dielectric_anisotropy_set(double e) {

  epsilon_ = (1.0/(12.0*pi_))*e;

  return;
}

/*****************************************************************************
 *
 *  blue_phase_dielectric_anisotropy
 *
 *
 *****************************************************************************/


__targetHost__ double blue_phase_dielectric_anisotropy(void) {

  return epsilon_;
}


/*****************************************************************************
 *
 *  blue_phase_set_active_region_gamma_zeta
 *
 *  Set the parameters gamma_ and zeta_ for inside and outside 
 *
 *  the active region.
 *****************************************************************************/

__targetHost__ void blue_phase_set_active_region_gamma_zeta(const int index) {
  
  double zeta_inside=0.0;
  double zeta_outside=0.0;
  
  double gamma_inside=3.0;
  double gamma_outside=2.4;

  /* check if we are inside/outside the active region */

  if ( coords_active_region(index) > 0.5 ){
    /*inside*/
    blue_phase_set_zeta(zeta_inside);
    blue_phase_set_gamma(gamma_inside);
  }
  else {
    /*outside*/
    blue_phase_set_zeta(zeta_outside);
    blue_phase_set_gamma(gamma_outside);
  }
  return;
}

/*****************************************************************************
 *
 *  fed_io_info
 *
 *****************************************************************************/

__targetHost__ int fed_io_info(io_info_t ** info) {

  assert(info);

  *info = io_info_fed;

  return 0;
}

/*****************************************************************************
 *
 *  fed_io_info_set
 *
 *****************************************************************************/

__targetHost__ int fed_io_info_set(io_info_t * info) {

  assert(info);
  io_info_fed = info;

  io_info_set_name(io_info_fed, "Free energy density");
  io_info_write_set(io_info_fed, IO_FORMAT_BINARY, fed_write);
  io_info_write_set(io_info_fed, IO_FORMAT_ASCII, fed_write_ascii);
  io_info_set_bytesize(io_info_fed, 3*sizeof(double));
 
  io_info_format_out_set(io_info_fed, IO_FORMAT_BINARY);
  io_info_metadata_filestub_set(io_info_fed, "fed");

  return 0;
}

/*****************************************************************************
 *
 *  fed_write_ascii
 *
 *  The "self" pointer is not required here.
 *
 *****************************************************************************/

static int fed_write_ascii(FILE * fp, int index, void * self) {


  int n;

  double q[3][3], dq[3][3][3];
  double fed[3];

  assert(fp);

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);


  //we are doing this on the host
  blue_phase_set_kernel_constants();
  void* pcon=NULL;
  blue_phase_host_constant_ptr(&pcon);


  fed[0] = blue_phase_compute_fed(q, dq, (bluePhaseKernelConstants_t*) pcon);
  fed[1] = blue_phase_compute_bulk_fed(q);
  fed[2] = blue_phase_compute_gradient_fed(q, dq);

  n = fprintf(fp, "%22.15e  %22.15e  %22.15e\n", fed[0], fed[1], fed[2]);
  if (n < 0) fatal("fprintf(fed) failed at index %d\n", index);

  return n;

}

/*****************************************************************************
 *
 *  fed_write
 *
 *  The "self" object is not required.
 *
 *****************************************************************************/

static int fed_write(FILE * fp, int index, void * self) {

  int n;

  double q[3][3], dq[3][3][3];
  double fed[3];

  assert(fp);

  field_tensor(q_, index, q);
  field_grad_tensor_grad(grad_q_, index, dq);


  //we are doing this on the host
  blue_phase_set_kernel_constants();
  void* pcon=NULL;
  blue_phase_host_constant_ptr(&pcon);

  fed[0] = blue_phase_compute_fed(q, dq, (bluePhaseKernelConstants_t*) pcon);
  fed[1] = blue_phase_compute_bulk_fed(q);
  fed[2] = blue_phase_compute_gradient_fed(q, dq);

  n = fwrite(fed, sizeof(double), 3, fp);
  if (n != 3) fatal("fwrite(fed) failed at index %d\n", index);

  return n;


}

/*****************************************************************************
 *
 *  blue_phase_scalar_ops
 *
 *  For symmetric traceless q[3][3], return the associated scalar
 *  order parameter, biaxial order parameter and director:
 *
 *  qs[0]  scalar order parameter: largest eigenvalue
 *  qs[1]  director[X] (associated eigenvector)
 *  qs[2]  director[Y]
 *  qs[3]  director[Z]
 *  qs[4]  biaxial order parameter b = sqrt(1 - 6 (Tr(QQQ))^2 / Tr(QQ)^3)
 *         related to the two largest eigenvalues...
 *
 *  If we write Q = ((s, 0, 0), (0, t, 0), (0, 0, -s -t)) then
 *
 *    Tr(QQ)  = s^2 + t^2 + (s + t)^2
 *    Tr(QQQ) = 3 s t (s + t)
 *
 *  If no diagonalisation is possible, all the results are set to zero.
 *
 *****************************************************************************/

__targetHost__ int blue_phase_scalar_ops(double q[3][3], double qs[5]) {

  int ifail;
  double eigenvalue[3];
  double eigenvector[3][3];
  double s, t;
  double q2, q3;

  ifail = util_jacobi_sort(q, eigenvalue, eigenvector);

  qs[0] = 0.0; qs[1] = 0.0; qs[2] = 0.0; qs[3] = 0.0; qs[4] = 0.0;

  if (ifail == 0) {

    qs[0] = eigenvalue[0];
    qs[1] = eigenvector[X][0];
    qs[2] = eigenvector[Y][0];
    qs[3] = eigenvector[Z][0];

    s = eigenvalue[0];
    t = eigenvalue[1];

    q2 = s*s + t*t + (s + t)*(s + t);
    q3 = 3.0*s*t*(s + t);
    qs[4] = sqrt(1 - 6.0*q3*q3 / (q2*q2*q2));
  }

  return ifail;
}
