/*****************************************************************************
 *
 *  phi_force.c
 *
 *  Computes the force on the fluid from the thermodynamic sector
 *  via the divergence of the chemical stress. Its calculation as
 *  a divergence ensures momentum is conserved.
 *
 *  Note that the stress may be asymmetric.
 *
 *  $Id: phi_force.c 1728 2012-07-18 08:41:51Z agray3 $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h> 

#define INCLUDING_FROM_GPU_SOURCE
#include "phi_force_gpu.h"

#include "pe.h"
//#include "coords.h"
#include "lattice.h"
#include "phi.h"
#include "site_map.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "wall.h"
#include "phi_force_stress.h"

// FROM util.c
#include "util.h"
//static const double r3_ = (1.0/3.0);


__constant__ double electric_cd[3];
__constant__ int nop_cd;
//__constant__ int N_cd[3];
//__constant__ int Nall_cd[3];
//__constant__ int nhalo_cd;
//__constant__ int nsites_cd;
__constant__ double redshift_cd;
__constant__ double rredshift_cd;
__constant__ double q0shift_cd;
__constant__ double a0_cd;
__constant__ double kappa0shift_cd;
__constant__ double kappa1shift_cd;
__constant__ double xi_cd;
__constant__ double zeta_cd;
__constant__ double gamma_cd;
__constant__ double epsilon_cd;
__constant__ double r3_cd;
__constant__ double d_cd[3][3];
__constant__ double e_cd[3][3][3];


extern "C" void checkCUDAError(const char *msg);

/*****************************************************************************
 *
 *  phi_force_calculation
 *
 *  Driver routine to compute the body force on fluid from phi sector.
 *
 *****************************************************************************/

void phi_force_calculation_gpu(void) {

  int N[3],nhalo,Nall[3];
  
  nhalo = coords_nhalo();
  coords_nlocal(N); 


  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  
  int nsites=Nall[X]*Nall[Y]*Nall[Z];
 

  

  // FROM blue_phase.c
  double q0_;        /* Pitch = 2pi / q0_ */
  double a0_;        /* Bulk free energy parameter A_0 */
  double gamma_;     /* Controls magnitude of order */
  double kappa0_;    /* Elastic constant \kappa_0 */
  double kappa1_;    /* Elastic constant \kappa_1 */
  
  double xi_;        /* effective molecular aspect ratio (<= 1.0) */
  double redshift_;  /* redshift parameter */
  double rredshift_; /* reciprocal */
  double zeta_;      /* Apolar activity parameter \zeta */
  
  double epsilon_; /* Dielectric anisotropy (e/12pi) */
  
  double electric_[3]; /* Electric field */
  


  redshift_ = blue_phase_redshift(); 
  rredshift_ = blue_phase_rredshift(); 
  q0_=blue_phase_q0();
  a0_=blue_phase_a0();
  kappa0_=blue_phase_kappa0();
  kappa1_=blue_phase_kappa1();
  xi_=blue_phase_get_xi();
  zeta_=blue_phase_get_zeta();
  gamma_=blue_phase_gamma();
  blue_phase_get_electric_field(electric_);
  epsilon_=blue_phase_get_dielectric_anisotropy();

 q0_ = q0_*rredshift_;
 kappa0_ = kappa0_*redshift_*redshift_;
 kappa1_ = kappa1_*redshift_*redshift_;


  //cudaMemcpy(electric_d, electric_, 3*sizeof(double), cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(electric_cd, electric_, 3*sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(N_cd, N, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(Nall_cd, Nall, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(nhalo_cd, &nhalo, sizeof(int), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(nsites_cd, &nsites, sizeof(int), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(redshift_cd, &redshift_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(rredshift_cd, &rredshift_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(q0shift_cd, &q0_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(a0_cd, &a0_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(kappa0shift_cd, &kappa0_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(kappa1shift_cd, &kappa1_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(xi_cd, &xi_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(zeta_cd, &zeta_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(gamma_cd, &gamma_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(epsilon_cd, &epsilon_, sizeof(double), 0, cudaMemcpyHostToDevice);
 cudaMemcpyToSymbol(r3_cd, &r3_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(d_cd, d_, 3*3*sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(e_cd, e_, 3*3*3*sizeof(double), 0, cudaMemcpyHostToDevice); 

 
  checkCUDAError("phi_force cudaMemcpyToSymbol");

  //if (force_required_ == 0) return;

  //if (le_get_nplane_total() > 0 || wall_present()) {
    /* Must use the flux method for LE planes */
    /* Also convenient for plane walls */
    //phi_force_flux();
  //}
  //else {
  //if (force_divergence_) {


  #define TPB 256
      int nblocks=(N[X]*N[Y]*N[Z]+TPB-1)/TPB;

      phi_force_calculation_fluid_gpu_d<<<nblocks,TPB>>>
	(le_index_real_to_buffer_d,phi_site_d,grad_phi_site_d,delsq_phi_site_d,force_d);
      
      cudaThreadSynchronize();
      checkCUDAError("phi_force_calculation_fluid_gpu_d");

      //}
      //else {
     //hi_force_fluid_phi_gradmu();
      //}
      //}

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


__device__ void blue_phase_compute_h_gpu_d(double q[3][3], double dq[3][3][3],
			      double dsq[3][3], double h[3][3]) {
  int ia, ib, ic, id;

  double q2;
  double e2;
  double eq;
  double sum;


  /* From the bulk terms in the free energy... */

  /* q2 = 0.0; */

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
      h[ia][ib] = -a0_cd*(1.0 - r3_cd*gamma_cd)*q[ia][ib]
  	+ a0_cd*gamma_cd*(sum - r3_cd*q2*d_cd[ia][ib]) - a0_cd*gamma_cd*q2*q[ia][ib];
    }
  }

  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */

  eq = 0.0;
  for (ib = 0; ib < 3; ib++) {
    for (ic = 0; ic < 3; ic++) {
      for (ia = 0; ia < 3; ia++) {
  	eq += e_cd[ib][ic][ia]*dq[ib][ic][ia];
      }
    }
  }


  /* d_c Q_db written as d_c Q_bd etc */
  //for (ia = 0; ia < 3; ia++) {
    //for (ib = 0; ib < 3; ib++) {
      /* sum = 0.0; */
      /* for (ic = 0; ic < 3; ic++) { */
      /* 	for (id = 0; id < 3; id++) { */
      /* 	  sum += */
      /* 	    (e_cd[ia][ic][id]*dq[ic][ib][id] + e_cd[ib][ic][id]*dq[ic][ia][id]); */
      /* 	} */
      /* } */
      /* h[ia][ib] +=  kappa0shift_cd*dsq[ia][ib] */
      /* 	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[ia][ib] */
      /* 	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[ia][ib]; */

      // }
  //}

      sum = 0.0;
      sum += (e_cd[0][0][0]*dq[0][0][0] + e_cd[0][0][0]*dq[0][0][0]);
      sum += (e_cd[0][0][1]*dq[0][0][1] + e_cd[0][0][1]*dq[0][0][1]);
      sum += (e_cd[0][0][2]*dq[0][0][2] + e_cd[0][0][2]*dq[0][0][2]);
      sum += (e_cd[0][1][0]*dq[1][0][0] + e_cd[0][1][0]*dq[1][0][0]);
      sum += (e_cd[0][1][1]*dq[1][0][1] + e_cd[0][1][1]*dq[1][0][1]);
      sum += (e_cd[0][1][2]*dq[1][0][2] + e_cd[0][1][2]*dq[1][0][2]);
      sum += (e_cd[0][2][0]*dq[2][0][0] + e_cd[0][2][0]*dq[2][0][0]);
      sum += (e_cd[0][2][1]*dq[2][0][1] + e_cd[0][2][1]*dq[2][0][1]);
      sum += (e_cd[0][2][2]*dq[2][0][2] + e_cd[0][2][2]*dq[2][0][2]);

      h[0][0] +=  kappa0shift_cd*dsq[0][0]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[0][0]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[0][0];

      sum = 0.0;
      sum += (e_cd[0][0][0]*dq[0][1][0] + e_cd[1][0][0]*dq[0][0][0]);
      sum += (e_cd[0][0][1]*dq[0][1][1] + e_cd[1][0][1]*dq[0][0][1]);
      sum += (e_cd[0][0][2]*dq[0][1][2] + e_cd[1][0][2]*dq[0][0][2]);
      sum += (e_cd[0][1][0]*dq[1][1][0] + e_cd[1][1][0]*dq[1][0][0]);
      sum += (e_cd[0][1][1]*dq[1][1][1] + e_cd[1][1][1]*dq[1][0][1]);
      sum += (e_cd[0][1][2]*dq[1][1][2] + e_cd[1][1][2]*dq[1][0][2]);
      sum += (e_cd[0][2][0]*dq[2][1][0] + e_cd[1][2][0]*dq[2][0][0]);
      sum += (e_cd[0][2][1]*dq[2][1][1] + e_cd[1][2][1]*dq[2][0][1]);
      sum += (e_cd[0][2][2]*dq[2][1][2] + e_cd[1][2][2]*dq[2][0][2]);

      h[0][1] +=  kappa0shift_cd*dsq[0][1]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[0][1]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[0][1];

      sum = 0.0;
      sum += (e_cd[0][0][0]*dq[0][2][0] + e_cd[2][0][0]*dq[0][0][0]);
      sum += (e_cd[0][0][1]*dq[0][2][1] + e_cd[2][0][1]*dq[0][0][1]);
      sum += (e_cd[0][0][2]*dq[0][2][2] + e_cd[2][0][2]*dq[0][0][2]);
      sum += (e_cd[0][1][0]*dq[1][2][0] + e_cd[2][1][0]*dq[1][0][0]);
      sum += (e_cd[0][1][1]*dq[1][2][1] + e_cd[2][1][1]*dq[1][0][1]);
      sum += (e_cd[0][1][2]*dq[1][2][2] + e_cd[2][1][2]*dq[1][0][2]);
      sum += (e_cd[0][2][0]*dq[2][2][0] + e_cd[2][2][0]*dq[2][0][0]);
      sum += (e_cd[0][2][1]*dq[2][2][1] + e_cd[2][2][1]*dq[2][0][1]);
      sum += (e_cd[0][2][2]*dq[2][2][2] + e_cd[2][2][2]*dq[2][0][2]);

      h[0][2] +=  kappa0shift_cd*dsq[0][2]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[0][2]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[0][2];

      ////////
      sum = 0.0;
      sum += (e_cd[1][0][0]*dq[0][0][0] + e_cd[0][0][0]*dq[0][1][0]);
      sum += (e_cd[1][0][1]*dq[0][0][1] + e_cd[0][0][1]*dq[0][1][1]);
      sum += (e_cd[1][0][2]*dq[0][0][2] + e_cd[0][0][2]*dq[0][1][2]);
      sum += (e_cd[1][1][0]*dq[1][0][0] + e_cd[0][1][0]*dq[1][1][0]);
      sum += (e_cd[1][1][1]*dq[1][0][1] + e_cd[0][1][1]*dq[1][1][1]);
      sum += (e_cd[1][1][2]*dq[1][0][2] + e_cd[0][1][2]*dq[1][1][2]);
      sum += (e_cd[1][2][0]*dq[2][0][0] + e_cd[0][2][0]*dq[2][1][0]);
      sum += (e_cd[1][2][1]*dq[2][0][1] + e_cd[0][2][1]*dq[2][1][1]);
      sum += (e_cd[1][2][2]*dq[2][0][2] + e_cd[0][2][2]*dq[2][1][2]);

      h[1][0] +=  kappa0shift_cd*dsq[1][0]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[1][0]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[1][0];

      sum = 0.0;
      sum += (e_cd[1][0][0]*dq[0][1][0] + e_cd[1][0][0]*dq[0][1][0]);
      sum += (e_cd[1][0][1]*dq[0][1][1] + e_cd[1][0][1]*dq[0][1][1]);
      sum += (e_cd[1][0][2]*dq[0][1][2] + e_cd[1][0][2]*dq[0][1][2]);
      sum += (e_cd[1][1][0]*dq[1][1][0] + e_cd[1][1][0]*dq[1][1][0]);
      sum += (e_cd[1][1][1]*dq[1][1][1] + e_cd[1][1][1]*dq[1][1][1]);
      sum += (e_cd[1][1][2]*dq[1][1][2] + e_cd[1][1][2]*dq[1][1][2]);
      sum += (e_cd[1][2][0]*dq[2][1][0] + e_cd[1][2][0]*dq[2][1][0]);
      sum += (e_cd[1][2][1]*dq[2][1][1] + e_cd[1][2][1]*dq[2][1][1]);
      sum += (e_cd[1][2][2]*dq[2][1][2] + e_cd[1][2][2]*dq[2][1][2]);

      h[1][1] +=  kappa0shift_cd*dsq[1][1]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[1][1]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[1][1];

      sum = 0.0;
      sum += (e_cd[1][0][0]*dq[0][2][0] + e_cd[2][0][0]*dq[0][1][0]);
      sum += (e_cd[1][0][1]*dq[0][2][1] + e_cd[2][0][1]*dq[0][1][1]);
      sum += (e_cd[1][0][2]*dq[0][2][2] + e_cd[2][0][2]*dq[0][1][2]);
      sum += (e_cd[1][1][0]*dq[1][2][0] + e_cd[2][1][0]*dq[1][1][0]);
      sum += (e_cd[1][1][1]*dq[1][2][1] + e_cd[2][1][1]*dq[1][1][1]);
      sum += (e_cd[1][1][2]*dq[1][2][2] + e_cd[2][1][2]*dq[1][1][2]);
      sum += (e_cd[1][2][0]*dq[2][2][0] + e_cd[2][2][0]*dq[2][1][0]);
      sum += (e_cd[1][2][1]*dq[2][2][1] + e_cd[2][2][1]*dq[2][1][1]);
      sum += (e_cd[1][2][2]*dq[2][2][2] + e_cd[2][2][2]*dq[2][1][2]);

      h[1][2] +=  kappa0shift_cd*dsq[1][2]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[1][2]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[1][2];

      /////
      sum = 0.0;
      sum += (e_cd[2][0][0]*dq[0][0][0] + e_cd[0][0][0]*dq[0][2][0]);
      sum += (e_cd[2][0][1]*dq[0][0][1] + e_cd[0][0][1]*dq[0][2][1]);
      sum += (e_cd[2][0][2]*dq[0][0][2] + e_cd[0][0][2]*dq[0][2][2]);
      sum += (e_cd[2][1][0]*dq[1][0][0] + e_cd[0][1][0]*dq[1][2][0]);
      sum += (e_cd[2][1][1]*dq[1][0][1] + e_cd[0][1][1]*dq[1][2][1]);
      sum += (e_cd[2][1][2]*dq[1][0][2] + e_cd[0][1][2]*dq[1][2][2]);
      sum += (e_cd[2][2][0]*dq[2][0][0] + e_cd[0][2][0]*dq[2][2][0]);
      sum += (e_cd[2][2][1]*dq[2][0][1] + e_cd[0][2][1]*dq[2][2][1]);
      sum += (e_cd[2][2][2]*dq[2][0][2] + e_cd[0][2][2]*dq[2][2][2]);

      h[2][0] +=  kappa0shift_cd*dsq[2][0]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[2][0]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[2][0];

      sum = 0.0;
      sum += (e_cd[2][0][0]*dq[0][1][0] + e_cd[1][0][0]*dq[0][2][0]);
      sum += (e_cd[2][0][1]*dq[0][1][1] + e_cd[1][0][1]*dq[0][2][1]);
      sum += (e_cd[2][0][2]*dq[0][1][2] + e_cd[1][0][2]*dq[0][2][2]);
      sum += (e_cd[2][1][0]*dq[1][1][0] + e_cd[1][1][0]*dq[1][2][0]);
      sum += (e_cd[2][1][1]*dq[1][1][1] + e_cd[1][1][1]*dq[1][2][1]);
      sum += (e_cd[2][1][2]*dq[1][1][2] + e_cd[1][1][2]*dq[1][2][2]);
      sum += (e_cd[2][2][0]*dq[2][1][0] + e_cd[1][2][0]*dq[2][2][0]);
      sum += (e_cd[2][2][1]*dq[2][1][1] + e_cd[1][2][1]*dq[2][2][1]);
      sum += (e_cd[2][2][2]*dq[2][1][2] + e_cd[1][2][2]*dq[2][2][2]);

      h[2][1] +=  kappa0shift_cd*dsq[2][1]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[2][1]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[2][1];

      sum = 0.0;
      sum += (e_cd[2][0][0]*dq[0][2][0] + e_cd[2][0][0]*dq[0][2][0]);
      sum += (e_cd[2][0][1]*dq[0][2][1] + e_cd[2][0][1]*dq[0][2][1]);
      sum += (e_cd[2][0][2]*dq[0][2][2] + e_cd[2][0][2]*dq[0][2][2]);
      sum += (e_cd[2][1][0]*dq[1][2][0] + e_cd[2][1][0]*dq[1][2][0]);
      sum += (e_cd[2][1][1]*dq[1][2][1] + e_cd[2][1][1]*dq[1][2][1]);
      sum += (e_cd[2][1][2]*dq[1][2][2] + e_cd[2][1][2]*dq[1][2][2]);
      sum += (e_cd[2][2][0]*dq[2][2][0] + e_cd[2][2][0]*dq[2][2][0]);
      sum += (e_cd[2][2][1]*dq[2][2][1] + e_cd[2][2][1]*dq[2][2][1]);
      sum += (e_cd[2][2][2]*dq[2][2][2] + e_cd[2][2][2]*dq[2][2][2]);

      h[2][2] +=  kappa0shift_cd*dsq[2][2]
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[2][2]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*q[2][2];






  /* Electric field term */

  e2 = 0.0;
  for (ia = 0; ia < 3; ia++) {
    e2 += electric_cd[ia]*electric_cd[ia];
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] +=  epsilon_cd*(electric_cd[ia]*electric_cd[ib] - r3_cd*d_cd[ia][ib]*e2);
    }
  }

  return;
}



/*****************************************************************************
 *
 *  blue_phase_compute_fed
 *
 *  Compute the free energy density as a function of q and the q gradient
 *  tensor dq.
 *
 *****************************************************************************/

__device__ double blue_phase_compute_fed_gpu_d(double q[3][3], double dq[3][3][3]){

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
	  sum += e_cd[ia][ic][id]*dq[ic][ib][id];
	}
      }
      sum += 2.0*q0shift_cd*q[ia][ib];
      dq1 += sum*sum;
    }
  }

  /* Electric field term (epsilon_ includes the factor 1/12pi) */

  efield = 0.0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      efield += electric_cd[ia]*q[ia][ib]*electric_cd[ib];
    }
  }

  sum = 0.5*a0_cd*(1.0 - r3_cd*gamma_cd)*q2 - r3_cd*a0_cd*gamma_cd*q3 +
    0.25*a0_cd*gamma_cd*q2*q2 + 0.5*kappa0shift_cd*dq0 + 0.5*kappa1shift_cd*dq1 - epsilon_cd*efield;;

  return sum;
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

__device__ void blue_phase_compute_stress_gpu_d(double q[3][3], double dq[3][3][3],
				   double h[3][3], double sth[3][3]){
  int ia, ib, ic, id, ie;

  double tmpdbl,tmpdbl2;
  
  /* We have ignored the rho T term at the moment, assumed to be zero
   * (in particular, it has no divergence if rho = const). */

  tmpdbl = 0.0 - blue_phase_compute_fed_gpu_d(q, dq);

  /* The contraction Q_ab H_ab */

  tmpdbl2 = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      tmpdbl2 += q[ia][ib]*h[ia][ib];
    }
  }

  /* The term in the isotropic pressure, plus that in qh */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sth[ia][ib] = -tmpdbl*d_cd[ia][ib] + 2.0*xi_cd*(q[ia][ib] + r3_cd*d_cd[ia][ib])*tmpdbl2;
    }
  }

  /* Remaining two terms in xi and molecular field */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      tmpdbl=0.;
      for (ic = 0; ic < 3; ic++) {
	tmpdbl+=
  	  -xi_cd*h[ia][ic]*(q[ib][ic] + r3_cd*d_cd[ib][ic])
  	  -xi_cd*(q[ia][ic] + r3_cd*d_cd[ia][ic])*h[ib][ic];
      }
      sth[ia][ib] += tmpdbl;
    }
  }

  /* Dot product term d_a Q_cd . dF/dQ_cd,b */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      tmpdbl=0.;
      for (ic = 0; ic < 3; ic++) {
	for (id = 0; id < 3; id++) {
	  tmpdbl +=
	    - kappa0shift_cd*dq[ia][ib][ic]*dq[id][ic][id]
	    - kappa1shift_cd*dq[ia][ic][id]*dq[ib][ic][id]
	    + kappa1shift_cd*dq[ia][ic][id]*dq[ic][ib][id];
	  
	  tmpdbl2= -2.0*kappa1shift_cd*q0shift_cd*dq[ia][ic][id];
	  for (ie = 0; ie < 3; ie++) {
	    tmpdbl +=
	     tmpdbl2*e_cd[ib][ic][ie]*q[id][ie];
	  }
	}
      }
      sth[ia][ib]+=tmpdbl;
    }
  }

  /* The antisymmetric piece q_ac h_cb - h_ac q_cb. We can
   * rewrite it as q_ac h_bc - h_ac q_bc. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      tmpdbl=0.;
      for (ic = 0; ic < 3; ic++) {
  	 tmpdbl += q[ia][ic]*h[ib][ic] - h[ia][ic]*q[ib][ic];
      }
      sth[ia][ib]+=tmpdbl;
      /* This is the minus sign. */
      sth[ia][ib] = -sth[ia][ib];
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

__device__ void blue_phase_chemical_stress_gpu_d(int index,
						 int *le_index_real_to_buffer_d,
						 double *phi_site_d,
						 double *grad_phi_site_d,
						 double *delsq_phi_site_d,
						 double sth[3][3]){


  int ia;

  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  //  if(threadIdx.x==0 && blockIdx.x==0) printf("in BPCS\n");

/*   phi_get_q_tensor(index, q); */
/*   phi_gradients_tensor_gradient(index, dq); */
/*   phi_gradients_tensor_delsq(index, dsq); */


  /* load phi */

  q[X][X] = phi_site_d[nsites_cd*XX+index];
  q[X][Y] = phi_site_d[nsites_cd*XY+index];
  q[X][Z] = phi_site_d[nsites_cd*XZ+index];
  q[Y][X] = q[X][Y];
  q[Y][Y] = phi_site_d[nsites_cd*YY+index];
  q[Y][Z] = phi_site_d[nsites_cd*YZ+index];
  q[Z][X] = q[X][Z];
  q[Z][Y] = q[Y][Z];
  q[Z][Z] = 0.0 - q[X][X] - q[Y][Y];


  /* load grad phi */
  for (ia = 0; ia < 3; ia++) {
    dq[ia][X][X] = grad_phi_site_d[ia*nsites_cd*5 + XX*nsites_cd + index];
    dq[ia][X][Y] = grad_phi_site_d[ia*nsites_cd*5 + XY*nsites_cd + index];
    dq[ia][X][Z] = grad_phi_site_d[ia*nsites_cd*5 + XZ*nsites_cd + index];
    dq[ia][Y][X] = dq[ia][X][Y];
    dq[ia][Y][Y] = grad_phi_site_d[ia*nsites_cd*5 + YY*nsites_cd + index];
    dq[ia][Y][Z] = grad_phi_site_d[ia*nsites_cd*5 + YZ*nsites_cd + index];
    dq[ia][Z][X] = dq[ia][X][Z];
    dq[ia][Z][Y] = dq[ia][Y][Z];
    dq[ia][Z][Z] = 0.0 - dq[ia][X][X] - dq[ia][Y][Y];
  }


    /* load delsq phi */
  dsq[X][X] = delsq_phi_site_d[XX*nsites_cd+index];
  dsq[X][Y] = delsq_phi_site_d[XY*nsites_cd+index];
  dsq[X][Z] = delsq_phi_site_d[XZ*nsites_cd+index];
  dsq[Y][X] = dsq[X][Y];
  dsq[Y][Y] = delsq_phi_site_d[YY*nsites_cd+index];
  dsq[Y][Z] = delsq_phi_site_d[YZ*nsites_cd+index];
  dsq[Z][X] = dsq[X][Z];
  dsq[Z][Y] = dsq[Y][Z];
  dsq[Z][Z] = 0.0 - dsq[X][X] - dsq[Y][Y];


  //DEV
 
  int i,j,k,icm1,icp1,indexm1,indexp1;
  get_coords_from_index_gpu_d(&i,&j,&k,index,Nall_cd);
  icm1=le_index_real_to_buffer_d[i];
  icp1=le_index_real_to_buffer_d[Nall_cd[X]+i];      

  indexm1 = get_linear_index_gpu_d(icm1,j,k,Nall_cd);
  indexp1 = get_linear_index_gpu_d(icp1,j,k,Nall_cd);

     dq[0][X][X]
       = 0.5*(phi_site_d[nsites_cd*XX+indexp1] - phi_site_d[nsites_cd*XX+indexm1]);
     dq[0][X][Y]
       = 0.5*(phi_site_d[nsites_cd*XY+indexp1] - phi_site_d[nsites_cd*XY+indexm1]);
     dq[0][X][Z]
       = 0.5*(phi_site_d[nsites_cd*XZ+indexp1] - phi_site_d[nsites_cd*XZ+indexm1]);
     dq[0][Y][X]
       = 0.5*(phi_site_d[nsites_cd*XY+indexp1] - phi_site_d[nsites_cd*XY+indexm1]);
     dq[0][Y][Y]
       = 0.5*(phi_site_d[nsites_cd*YY+indexp1] - phi_site_d[nsites_cd*YY+indexm1]);
     dq[0][Y][Z]
       = 0.5*(phi_site_d[nsites_cd*YZ+indexp1] - phi_site_d[nsites_cd*YZ+indexm1]);
     dq[0][Z][X]
       = 0.5*(phi_site_d[nsites_cd*XZ+indexp1] - phi_site_d[nsites_cd*XZ+indexm1]);
     dq[0][Z][Y]
       = 0.5*(phi_site_d[nsites_cd*YZ+indexp1] - phi_site_d[nsites_cd*YZ+indexm1]);
     dq[0][Z][Z]
       = 0. - 0.5*(phi_site_d[nsites_cd*XX+indexp1] - phi_site_d[nsites_cd*XX+indexm1])
       - 0.5*(phi_site_d[nsites_cd*YY+indexp1] - phi_site_d[nsites_cd*YY+indexm1]);

 //END DEV


     blue_phase_compute_h_gpu_d(q, dq, dsq, h);
     //blue_phase_compute_h_gpu_d_test2(q, dq, dsq, h);
  blue_phase_compute_stress_gpu_d(q, dq, h, sth);

  return;
}

/*****************************************************************************
 *
 *  phi_force_calculation_fluid
 *
 *  Compute force from thermodynamic sector via
 *    F_alpha = nalba_beta Pth_alphabeta
 *  using a simple six-point stencil.
 *
 *  Side effect: increments the force at each local lattice site in
 *  preparation for the collision stage.
 *
 *****************************************************************************/


__global__ void phi_force_calculation_fluid_gpu_d(int * le_index_real_to_buffer_d,
						  double *phi_site_d,
						  double *grad_phi_site_d,
						  double *delsq_phi_site_d,
						  double *force_d
					    ) {

  int ia, icm1, icp1;
  int index, index1;
  double pth0[3][3];
  double pth1[3][3];
  double force[3];
  int threadIndex,ii, jj, kk;

 /* CUDA thread index */
 threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
 
 /* Avoid going beyond problem domain */
 if (threadIndex < N_cd[X]*N_cd[Y]*N_cd[Z])
    {


      /* calculate index from CUDA thread index */

      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,N_cd);
      index = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);      
      icm1=le_index_real_to_buffer_d[ii+nhalo_cd];
      icp1=le_index_real_to_buffer_d[Nall_cd[X]+ii+nhalo_cd];      
      

	/* Compute pth at current point */
      blue_phase_chemical_stress_gpu_d(index,le_index_real_to_buffer_d,phi_site_d,grad_phi_site_d,delsq_phi_site_d,pth0);

	/* Compute differences */
	index1 = get_linear_index_gpu_d(icp1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,grad_phi_site_d,delsq_phi_site_d,pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	index1 = get_linear_index_gpu_d(icm1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,grad_phi_site_d,delsq_phi_site_d,pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	
	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd+1,kk+nhalo_cd,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
	

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd+1,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	/* Store the force on lattice */
	for (ia=0;ia<3;ia++)
	  force_d[ia*nsites_cd+index]+=force[ia];

    }


  return;
}


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N_d[3])
{
  
  int yfac = N_d[Z];
  int xfac = N_d[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}

/* get 3d coordinates from the index on the accelerator */
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N_d[3])

{
  
  int yfac = N_d[Z];
  int xfac = N_d[Y]*yfac;
  
  *ii = index/xfac;
  *jj = ((index-xfac*(*ii))/yfac);
  *kk = (index-(*ii)*xfac-(*jj)*yfac);

  return;

}
