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
#include "colloids_Q_tensor.h"
// FROM util.c
#include "util.h"
//static const double r3_ = (1.0/3.0);


__constant__ double electric_cd[3];
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
__constant__ double dt_solid_cd;
__constant__ double dt_cd;
__constant__ double Gamma_cd;

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

cudaMemcpyToSymbol(N_cd, N, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(Nall_cd, Nall, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nhalo_cd, &nhalo, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nsites_cd, &nsites, sizeof(int), 0, cudaMemcpyHostToDevice); 
 
  cudaMemcpyToSymbol(electric_cd, electric_, 3*sizeof(double), 0, cudaMemcpyHostToDevice); 
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


  cudaFuncSetCacheConfig(phi_force_calculation_fluid_gpu_d,cudaFuncCachePreferL1);

  // #define TPB 256
  //#define TPBX 4 
  //#define TPBY 4
  //#define TPBZ 8

  #define TPBX 4 
  #define TPBY 4
  #define TPBZ 8

    //int nblocks=(N[X]*N[Y]*N[Z]+TPB-1)/TPB;
  
  dim3 nblocks((N[Z]+TPBZ-1)/TPBZ,(N[Y]+TPBY-1)/TPBY,(N[X]+TPBX-1)/TPBX);
  dim3 threadsperblock(TPBZ,TPBY,TPBX);

  phi_force_calculation_fluid_gpu_d<<<nblocks,threadsperblock>>>
  //phi_force_calculation_fluid_gpu_d<<<nblocks,TPB>>>
    (le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,force_d);
      
      cudaThreadSynchronize();
      checkCUDAError("phi_force_calculation_fluid_gpu_d");

      //}
      //else {
     //hi_force_fluid_phi_gradmu();
      //}
      //}

  return;
}


void phi_force_colloid_gpu(void) {

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

cudaMemcpyToSymbol(N_cd, N, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(Nall_cd, Nall, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nhalo_cd, &nhalo, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nsites_cd, &nsites, sizeof(int), 0, cudaMemcpyHostToDevice); 
 
  cudaMemcpyToSymbol(electric_cd, electric_, 3*sizeof(double), 0, cudaMemcpyHostToDevice); 
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

 
  checkCUDAError("phi_force_colloid cudaMemcpyToSymbol");

  // TODO
  /* if (colloids_q_anchoring_method() == ANCHORING_METHOD_ONE) { */
  /*   phi_force_interpolation1(); */
  /* } */
  /* else { */
  /*   phi_force_interpolation2(); */
  /* } */


  cudaFuncSetCacheConfig(phi_force_colloid_gpu_d,cudaFuncCachePreferL1);

  #define TPBX 4 
  #define TPBY 4
  #define TPBZ 8

  
  dim3 nblocks((N[Z]+TPBZ-1)/TPBZ,(N[Y]+TPBY-1)/TPBY,(N[X]+TPBX-1)/TPBX);
  dim3 threadsperblock(TPBZ,TPBY,TPBX);

  phi_force_colloid_gpu_d<<<nblocks,threadsperblock>>>
    (le_index_real_to_buffer_d,site_map_status_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,force_d,colloid_force_d);
      
      cudaThreadSynchronize();
      checkCUDAError("phi_force_colloid_gpu_d");

  return;
}



void blue_phase_be_update_gpu(void) {

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


  int nop = phi_nop();
  assert(nop == 5);

  /* For first anchoring method (only) have evolution at solid sites. */
  const double dt = 1.0;
  double dt_solid;
  dt_solid = 0;
  if (colloids_q_anchoring_method() == ANCHORING_METHOD_ONE) dt_solid = dt;

  double Gamma_=blue_phase_be_get_rotational_diffusion();


  //cudaMemcpy(electric_d, electric_, 3*sizeof(double), cudaMemcpyHostToDevice); 

cudaMemcpyToSymbol(N_cd, N, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(Nall_cd, Nall, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nhalo_cd, &nhalo, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nsites_cd, &nsites, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nop_cd, &nop, sizeof(int), 0, cudaMemcpyHostToDevice); 
 
  cudaMemcpyToSymbol(electric_cd, electric_, 3*sizeof(double), 0, cudaMemcpyHostToDevice); 
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
 cudaMemcpyToSymbol(dt_solid_cd, &dt_solid, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(dt_cd, &dt, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(Gamma_cd, &Gamma_, sizeof(double), 0, cudaMemcpyHostToDevice); 

 
 checkCUDAError("blue_phase_be_update cudaMemcpyToSymbol");



  cudaFuncSetCacheConfig(phi_force_calculation_fluid_gpu_d,cudaFuncCachePreferL1);

  #define TPBX 4 
  #define TPBY 4
  #define TPBZ 8

  dim3 nblocks((N[Z]+TPBZ-1)/TPBZ,(N[Y]+TPBY-1)/TPBY,(N[X]+TPBX-1)/TPBX);
  dim3 threadsperblock(TPBZ,TPBY,TPBX);


  blue_phase_be_update_gpu_d<<<nblocks,threadsperblock>>>
    (le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,force_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d, hs5_d);
      
  cudaThreadSynchronize();
  checkCUDAError("blue_phase_be_update_gpu_d");


  return;
}

void advection_upwind_gpu(void) {

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


  int nop = phi_nop();
  assert(nop == 5);

  /* For first anchoring method (only) have evolution at solid sites. */
  const double dt = 1.0;
  double dt_solid;
  dt_solid = 0;
  if (colloids_q_anchoring_method() == ANCHORING_METHOD_ONE) dt_solid = dt;

  double Gamma_=blue_phase_be_get_rotational_diffusion();


  //cudaMemcpy(electric_d, electric_, 3*sizeof(double), cudaMemcpyHostToDevice); 

cudaMemcpyToSymbol(N_cd, N, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(Nall_cd, Nall, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nhalo_cd, &nhalo, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nsites_cd, &nsites, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nop_cd, &nop, sizeof(int), 0, cudaMemcpyHostToDevice); 
 
  cudaMemcpyToSymbol(electric_cd, electric_, 3*sizeof(double), 0, cudaMemcpyHostToDevice); 
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
 cudaMemcpyToSymbol(dt_solid_cd, &dt_solid, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(dt_cd, &dt, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(Gamma_cd, &Gamma_, sizeof(double), 0, cudaMemcpyHostToDevice); 

 
 checkCUDAError("advection_upwind cudaMemcpyToSymbol");



 cudaFuncSetCacheConfig(advection_upwind_gpu_d,cudaFuncCachePreferL1);

  #define TPBX 4 
  #define TPBY 4
  #define TPBZ 8

 /* two sfastest moving dimensions have cover a single lower width-one halo here */
 dim3 nblocks(((N[Z]+1)+TPBZ-1)/TPBZ,((N[Y]+1)+TPBY-1)/TPBY,(N[X]+TPBX-1)/TPBX);
  dim3 threadsperblock(TPBZ,TPBY,TPBX);

  advection_upwind_gpu_d<<<nblocks,threadsperblock>>>
    (le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,force_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d, hs5_d);
      
  cudaThreadSynchronize();
  checkCUDAError("advection_upwind_gpu_d");


  return;
}

__global__ void advection_bcs_no_normal_flux_gpu_d(const int nop,
					   char *site_map_status_d,
					   double *fluxe_d,
					   double *fluxw_d,
					   double *fluxy_d,
					   double *fluxz_d
						   );


void advection_bcs_no_normal_flux_gpu(void){

  int N[3],nhalo,Nall[3];
  
  nhalo = coords_nhalo();
  coords_nlocal(N); 


  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  
  int nsites=Nall[X]*Nall[Y]*Nall[Z];


  int nop = phi_nop();
  
  cudaFuncSetCacheConfig(advection_upwind_gpu_d,cudaFuncCachePreferL1);
  
#define TPBX 4 
#define TPBY 4
#define TPBZ 8
  
  /* two sfastest moving dimensions have cover a single lower width-one halo here */
  dim3 nblocks(((N[Z]+1)+TPBZ-1)/TPBZ,((N[Y]+1)+TPBY-1)/TPBY,(N[X]+TPBX-1)/TPBX);
  dim3 threadsperblock(TPBZ,TPBY,TPBX);
  
  advection_bcs_no_normal_flux_gpu_d<<<nblocks,threadsperblock>>>
    (nop, site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d);
  
  cudaThreadSynchronize();
  checkCUDAError("advection_bcs_no_normal_flux_gpu");
  
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


__device__ void blue_phase_compute_h_gpu_d(double h[3][3], double *phi_site_full_d, 
					   int index, int indexm1, int indexp1) {
  int ia, ib, ic;

  double q2;
  double e2;
  double eq;
  double sum;


  /* From the bulk terms in the free energy... */

  q2 = 0.0;
  eq = 0.0;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      q2 += phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index]*phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index];


      eq += e_cd[0][ib][ia]*0.5*(phi_site_full_d[3*nsites_cd*ib+nsites_cd*ia+indexp1] -
				 phi_site_full_d[3*nsites_cd*ib+nsites_cd*ia+indexm1]);
      eq += e_cd[1][ib][ia]*0.5*(phi_site_full_d[3*nsites_cd*ib+nsites_cd*ia+index+N_cd[Z]+2*nhalo_cd] -
				 phi_site_full_d[3*nsites_cd*ib+nsites_cd*ia+index-N_cd[Z]-2*nhalo_cd]);
      eq += e_cd[2][ib][ia]*0.5*(phi_site_full_d[3*nsites_cd*ib+nsites_cd*ia+index+1] -
				 phi_site_full_d[3*nsites_cd*ib+nsites_cd*ia+index-1]);


    }
  }

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {
  	sum +=  phi_site_full_d[3*nsites_cd*ia+nsites_cd*ic+index]* phi_site_full_d[3*nsites_cd*ib+nsites_cd*ic+index];
      }
      h[ia][ib] = -a0_cd*(1.0 - r3_cd*gamma_cd)* phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index]
  	+ a0_cd*gamma_cd*(sum - r3_cd*q2*d_cd[ia][ib]) - a0_cd*gamma_cd*q2*phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index];
    }
  }


  /* From the gradient terms ... */
  /* First, the sum e_abc d_b Q_ca. With two permutations, we
   * may rewrite this as e_bca d_b Q_ca */
  //now above
  //  eq = 0.0;
  //  for (ia = 0; ia < 3; ia++) {
  //	for (ib = 0; ib < 3; ib++) {
  // }
  //}




  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      for (ic = 0; ic < 3; ic++) {

      	  sum +=
      	    (e_cd[ia][0][ic]*0.5*(phi_site_full_d[3*nsites_cd*ib+nsites_cd*ic+indexp1] -
	  			  phi_site_full_d[3*nsites_cd*ib+nsites_cd*ic+indexm1]) +
	     e_cd[ib][0][ic]*0.5*(phi_site_full_d[3*nsites_cd*ia+nsites_cd*ic+indexp1] -
	  			  phi_site_full_d[3*nsites_cd*ia+nsites_cd*ic+indexm1]));
      	  sum +=
      	    (e_cd[ia][1][ic]* 0.5*(phi_site_full_d[3*nsites_cd*ib+nsites_cd*ic+index+N_cd[Z]+2*nhalo_cd] -
	  			   phi_site_full_d[3*nsites_cd*ib+nsites_cd*ic+index-N_cd[Z]-2*nhalo_cd])
	     + e_cd[ib][1][ic]* 0.5*(phi_site_full_d[3*nsites_cd*ia+nsites_cd*ic+index+N_cd[Z]+2*nhalo_cd] -
	  			     phi_site_full_d[3*nsites_cd*ia+nsites_cd*ic+index-N_cd[Z]-2*nhalo_cd]));
      	  sum +=
      	    (e_cd[ia][2][ic]* 0.5*(phi_site_full_d[3*nsites_cd*ib+nsites_cd*ic+index+1] -
	  			   phi_site_full_d[3*nsites_cd*ib+nsites_cd*ic+index-1])
	     + e_cd[ib][2][ic]* 0.5*(phi_site_full_d[3*nsites_cd*ia+nsites_cd*ic+index+1] -
	  			     phi_site_full_d[3*nsites_cd*ia+nsites_cd*ic+index-1]));
      }
      h[ia][ib] +=  kappa0shift_cd*
	(phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+indexp1] 
	+ phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+indexm1]
	+ phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index+N_cd[Z]+2*nhalo_cd] 
	+ phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index-N_cd[Z]-2*nhalo_cd]
	+ phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index+1] 
	+ phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index-1]
	 - 6*phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index])
      	- 2.0*kappa1shift_cd*q0shift_cd*sum + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[ia][ib]
      	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index];

      }
  }





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
						 double *phi_site_full_d,
						 double *grad_phi_site_d,
						 double *delsq_phi_site_d,
						 double sth[3][3]){


  int ia;

  double q[3][3];
  double h[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  //  if(threadIdx.x==0 && blockIdx.x==0) printf("in BPCS\n");

  //phi_get_q_tensor(index, q);
  //phi_gradients_tensor_gradient(index, dq);
  //phi_gradients_tensor_delsq(index, dsq);


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

  int i,j,k,icm1,icp1,indexm1,indexp1;
  get_coords_from_index_gpu_d(&i,&j,&k,index,Nall_cd);
  icm1=le_index_real_to_buffer_d[i];
  icp1=le_index_real_to_buffer_d[Nall_cd[X]+i];      

  indexm1 = get_linear_index_gpu_d(icm1,j,k,Nall_cd);
  indexp1 = get_linear_index_gpu_d(icp1,j,k,Nall_cd);

   blue_phase_compute_h_gpu_d(h, phi_site_full_d, index, indexm1, indexp1);
  //blue_phase_compute_h_gpu_d(sth, phi_site_full_d, index, indexm1, indexp1);
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
						  double *phi_site_full_d,
						  double *grad_phi_site_d,
						  double *delsq_phi_site_d,
						  double *force_d
					    ) {

  int ia, icm1, icp1;
  int index, index1;
  double pth0[3][3];
  double pth1[3][3];
  double force[3];
  int ii, jj, kk;
  //int threadIndex,ii, jj, kk;

 /* CUDA thread index */
  //threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  // if (threadIndex < N_cd[X]*N_cd[Y]*N_cd[Z])

  if (ii < N_cd[X] && jj < N_cd[Y] && kk < N_cd[Z] )
    {


      /* calculate index from CUDA thread index */

      //get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,N_cd);
      index = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);      
      icm1=le_index_real_to_buffer_d[ii+nhalo_cd];
      icp1=le_index_real_to_buffer_d[Nall_cd[X]+ii+nhalo_cd];      
      

	/* Compute pth at current point */
      blue_phase_chemical_stress_gpu_d(index,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,pth0);

	/* Compute differences */
	index1 = get_linear_index_gpu_d(icp1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	index1 = get_linear_index_gpu_d(icm1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	
	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd+1,kk+nhalo_cd,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
	

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd+1,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);
	blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	/* Store the force on lattice */
	for (ia=0;ia<3;ia++)
	  force_d[ia*nsites_cd+index]+=force[ia];

    }


  return;
}



__global__ void phi_force_colloid_gpu_d(int * le_index_real_to_buffer_d,
					char * site_map_status_d,
						  double *phi_site_d,
						  double *phi_site_full_d,
						  double *grad_phi_site_d,
						  double *delsq_phi_site_d,
					double *force_d,
						  double *colloid_force_d
					    ) {

  int ia, icm1, icp1;
  int index, index1;
  double pth0[3][3];
  double pth1[3][3];
  double force[3];
  int ii, jj, kk;
  //int threadIndex,ii, jj, kk;

 /* CUDA thread index */
  //threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  // if (threadIndex < N_cd[X]*N_cd[Y]*N_cd[Z])

  if (ii < N_cd[X] && jj < N_cd[Y] && kk < N_cd[Z] )
    {


      /* calculate index from CUDA thread index */

      //get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,N_cd);
      index = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);      

      if (site_map_status_d[index1] != COLLOID){
	/* If this is solid, then there's no contribution here. */
      
	/* Compute pth at current point */
      blue_phase_chemical_stress_gpu_d(index,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,pth0);

	/* Compute differences */
	index1 = get_linear_index_gpu_d(ii+nhalo_cd+1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);

	if (site_map_status_d[index1] == COLLOID){
	  /* Compute the fluxes at solid/fluid boundary */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] = -pth0[ia][X];
	    colloid_force_d[0*nsites_cd*3+nsites_cd*ia+index]+=pth0[ia][X];
	  }
	}
	else
	  {
	  /* This flux is fluid-fluid */ 

	    blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	    }

	  }



	index1 = get_linear_index_gpu_d(ii+nhalo_cd-1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);

	if (site_map_status_d[index1] == COLLOID){
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][X];
	    colloid_force_d[1*nsites_cd*3+nsites_cd*ia+index]-=pth0[ia][X];
	  }
	}
	else
	  {
	    /* Fluid - fluid */
	    
	    blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	    }
	  }

	
	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd+1,kk+nhalo_cd,Nall_cd);
	
	if (site_map_status_d[index1] == COLLOID){
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Y];
	    colloid_force_d[2*nsites_cd*3+nsites_cd*ia+index]+=pth0[ia][Y];
	  }
	}
	else
	  {
	    /* Fluid - fluid */
	    
	    blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	    }
	  }
	
	
	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);
	
	if (site_map_status_d[index1] == COLLOID){
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Y];
	    colloid_force_d[3*nsites_cd*3+nsites_cd*ia+index]-=pth0[ia][Y];
	  }
	}
	else
	  {
	    /* Fluid - fluid */
	    
	    blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	    }
	  }	
	
	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd+1,Nall_cd);
	
	if (site_map_status_d[index1] == COLLOID){
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] -= pth0[ia][Z];
	    colloid_force_d[4*nsites_cd*3+nsites_cd*ia+index]+=pth0[ia][Z];
	  }
	}
	else
	  {
	    /* Fluid - fluid */
	    
	    blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	    }
	    
	  }
	
	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);
	if (site_map_status_d[index1] == COLLOID){
	  /* Solid-fluid */
	  for (ia = 0; ia < 3; ia++) {
	    force[ia] += pth0[ia][Z];
	    colloid_force_d[5*nsites_cd*3+nsites_cd*ia+index]-=pth0[ia][Z];
	  }
	}
	else
	  {
	    /* Fluid - fluid */
	    	    
	    blue_phase_chemical_stress_gpu_d(index1,le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d, pth1);
	    for (ia = 0; ia < 3; ia++) {
	      force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	    }
	    
	  }

	/* Store the force on lattice */
	for (ia=0;ia<3;ia++)
	  force_d[ia*nsites_cd+index]+=force[ia];
	
      }
    }
  
  
  return;
}


__global__ void blue_phase_be_update_gpu_d(int * le_index_real_to_buffer_d,
						  double *phi_site_d,
						  double *phi_site_full_d,
						  double *grad_phi_site_d,
						  double *delsq_phi_site_d,
					   double *force_d,
					   double *velocity_d,
					   char *site_map_status_d,
					   double *fluxe_d,
					   double *fluxw_d,
					   double *fluxy_d,
					   double *fluxz_d,
					   double *hs5_d
					    ) {

  int ia, icm1, icp1;
  int index, index1, indexm1, indexp1;
  double pth0[3][3];
  double pth1[3][3];
  double force[3];
  int ii, jj, kk;
  //int threadIndex,ii, jj, kk;



 /* CUDA thread index */
  //threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  // if (threadIndex < N_cd[X]*N_cd[Y]*N_cd[Z])

  if (ii < N_cd[X] && jj < N_cd[Y] && kk < N_cd[Z] )
    {


      /* calculate index from CUDA thread index */

      //get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,N_cd);
      index = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);      

      icm1=le_index_real_to_buffer_d[ii+nhalo_cd];
      icp1=le_index_real_to_buffer_d[Nall_cd[X]+ii+nhalo_cd];      

      indexm1 = get_linear_index_gpu_d(icm1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);
      indexp1 = get_linear_index_gpu_d(icp1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);



      int ia, ib, ic, id;

  double q[3][3];
  double d[3][3];
  double h[3][3];
  double s[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  double w[3][3];
  double omega[3][3];
  double trace_qw;

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


   blue_phase_compute_h_gpu_d(h, phi_site_full_d, index, indexm1, indexp1);


   if (site_map_status_d[index] != FLUID) {
     
     q[X][X] += dt_solid_cd*Gamma_cd*h[X][X];
     q[X][Y] += dt_solid_cd*Gamma_cd*h[X][Y];
     q[X][Z] += dt_solid_cd*Gamma_cd*h[X][Z];
     q[Y][Y] += dt_solid_cd*Gamma_cd*h[Y][Y];
     q[Y][Z] += dt_solid_cd*Gamma_cd*h[Y][Z];
     
   }
   else {
     

	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

     // hydrodynamics_velocity_gradient_tensor(ic, jc, kc, w);	  

       w[X][X] = 0.5*(velocity_d[X*nsites_cd+indexp1] - velocity_d[X*nsites_cd+indexm1]);
       w[Y][X] = 0.5*(velocity_d[Y*nsites_cd+indexp1] - velocity_d[Y*nsites_cd+indexm1]);
       w[Z][X] = 0.5*(velocity_d[Z*nsites_cd+indexp1] - velocity_d[Z*nsites_cd+indexm1]);
       
       indexm1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);
       indexp1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd+1,kk+nhalo_cd,Nall_cd);

       w[X][Y] = 0.5*(velocity_d[X*nsites_cd+indexp1] - velocity_d[X*nsites_cd+indexm1]);
       w[Y][Y] = 0.5*(velocity_d[Y*nsites_cd+indexp1] - velocity_d[Y*nsites_cd+indexm1]);
       w[Z][Y] = 0.5*(velocity_d[Z*nsites_cd+indexp1] - velocity_d[Z*nsites_cd+indexm1]);

       indexm1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);
       indexp1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd+1,Nall_cd);

       w[X][Z] = 0.5*(velocity_d[X*nsites_cd+indexp1] - velocity_d[X*nsites_cd+indexm1]);
       w[Y][Z] = 0.5*(velocity_d[Y*nsites_cd+indexp1] - velocity_d[Y*nsites_cd+indexm1]);
       w[Z][Z] = 0.5*(velocity_d[Z*nsites_cd+indexp1] - velocity_d[Z*nsites_cd+indexm1]);
       
     //end  hydrodynamics_velocity_gradient_tensor(ic, jc, kc, w);
	  trace_qw = 0.0;

	  for (ia = 0; ia < 3; ia++) {
	    trace_qw += q[ia][ia]*w[ia][ia];
	    for (ib = 0; ib < 3; ib++) {
	      d[ia][ib]     = 0.5*(w[ia][ib] + w[ib][ia]);
	      omega[ia][ib] = 0.5*(w[ia][ib] - w[ib][ia]);
	    }
	  }
	  
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      s[ia][ib] = -2.0*xi_cd*(q[ia][ib] + r3_cd*d_cd[ia][ib])*trace_qw;
	      for (id = 0; id < 3; id++) {
		s[ia][ib] +=
		  (xi_cd*d[ia][id] + omega[ia][id])*(q[id][ib] + r3_cd*d_cd[id][ib])
		  + (q[ia][id] + r3_cd*d_cd[ia][id])*(xi_cd*d[id][ib] - omega[id][ib]);
	      }
	    }
	  }
	     
	  /* Here's the full hydrodynamic update. */
	  

	/*   indexj = le_site_index(ic, jc-1, kc); */
	/*   indexk = le_site_index(ic, jc, kc-1); */

	  int indexj, indexk;
	  indexj = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);      
	  indexk = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);      

	  q[X][X] += dt_cd*(s[X][X] + Gamma_cd*(h[X][X] + hs5_d[nop_cd*index + XX])
	  		 - fluxe_d[nop_cd*index + XX] + fluxw_d[nop_cd*index  + XX]
	  		 - fluxy_d[nop_cd*index + XX] + fluxy_d[nop_cd*indexj + XX]
	  		 - fluxz_d[nop_cd*index + XX] + fluxz_d[nop_cd*indexk + XX]);

	  q[X][Y] += dt_cd*(s[X][Y] + Gamma_cd*(h[X][Y] + hs5_d[nop_cd*index + XY])
	  		 - fluxe_d[nop_cd*index + XY] + fluxw_d[nop_cd*index  + XY]
	  		 - fluxy_d[nop_cd*index + XY] + fluxy_d[nop_cd*indexj + XY]
	  		 - fluxz_d[nop_cd*index + XY] + fluxz_d[nop_cd*indexk + XY]);

	  q[X][Z] += dt_cd*(s[X][Z] + Gamma_cd*(h[X][Z] + hs5_d[nop_cd*index + XZ])
	  		 - fluxe_d[nop_cd*index + XZ] + fluxw_d[nop_cd*index  + XZ]
	  		 - fluxy_d[nop_cd*index + XZ] + fluxy_d[nop_cd*indexj + XZ]
	  		 - fluxz_d[nop_cd*index + XZ] + fluxz_d[nop_cd*indexk + XZ]);

	  q[Y][Y] += dt_cd*(s[Y][Y] + Gamma_cd*(h[Y][Y] + hs5_d[nop_cd*index + YY])
	  		 - fluxe_d[nop_cd*index + YY] + fluxw_d[nop_cd*index  + YY]
	  		 - fluxy_d[nop_cd*index + YY] + fluxy_d[nop_cd*indexj + YY]
	  		 - fluxz_d[nop_cd*index + YY] + fluxz_d[nop_cd*indexk + YY]);

	  q[Y][Z] += dt_cd*(s[Y][Z] + Gamma_cd*(h[Y][Z] + hs5_d[nop_cd*index + YZ])
	  		 - fluxe_d[nop_cd*index + YZ] + fluxw_d[nop_cd*index  + YZ]
	  		 - fluxy_d[nop_cd*index + YZ] + fluxy_d[nop_cd*indexj + YZ]
	  		 - fluxz_d[nop_cd*index + YZ] + fluxz_d[nop_cd*indexk + YZ]);
	
	/* phi_set_q_tensor(index, q); */

	   }
	 phi_site_d[nsites_cd*XX+index] = q[X][X];
	 phi_site_d[nsites_cd*XY+index] = q[X][Y];
	 phi_site_d[nsites_cd*XZ+index] = q[X][Z];
	 phi_site_d[nsites_cd*YY+index] = q[Y][Y];
	 phi_site_d[nsites_cd*YZ+index] = q[Y][Z];


    }


  return;
}


__global__ void advection_upwind_gpu_d(int * le_index_real_to_buffer_d,
						  double *phi_site_d,
						  double *phi_site_full_d,
						  double *grad_phi_site_d,
						  double *delsq_phi_site_d,
					   double *force_d,
					   double *velocity_d,
					   char *site_map_status_d,
					   double *fluxe_d,
					   double *fluxw_d,
					   double *fluxy_d,
					   double *fluxz_d,
					   double *hs5_d
					    ) {

  int ia, icm1, icp1;
  int index, index1, index0;
  int ii, jj, kk, n;
  double u, phi0;
  //int threadIndex,ii, jj, kk;



 /* CUDA thread index */
  //threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  // if (threadIndex < N_cd[X]*N_cd[Y]*N_cd[Z])

  if (ii < N_cd[X] && jj < (N_cd[Y]+1) && kk < (N_cd[Z]+1) )
    {


      /* calculate index from CUDA thread index */

      //get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,N_cd);
      index0 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);      
      icm1=le_index_real_to_buffer_d[ii+nhalo_cd];
      icp1=le_index_real_to_buffer_d[Nall_cd[X]+ii+nhalo_cd];      


      
      for (n = 0; n < nop_cd; n++) {

	phi0 = phi_site_d[nsites_cd*n+index0];
	index1=get_linear_index_gpu_d(icm1,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);
	u = 0.5*(velocity_d[X*nsites_cd+index0] + velocity_d[X*nsites_cd+index1]);
	
	if (u > 0.0) {
	  fluxw_d[nop_cd*index0 + n] = u*phi_site_d[nsites_cd*n+index1];
	}
	else {
	  fluxw_d[nop_cd*index0 + n] = u*phi0;
	}

	  /* east face (ic and icp1) */

	index1=get_linear_index_gpu_d(icp1,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);
	  u = 0.5*(velocity_d[X*nsites_cd+index0] + velocity_d[X*nsites_cd+index1]);

	  if (u < 0.0) {
	    fluxe_d[nop_cd*index0 + n] = u*phi_site_d[nsites_cd*n+index1];
	  }
	  else {
	    fluxe_d[nop_cd*index0 + n] = u*phi0;
	  }


	  /* y direction */

	index1=get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);
	  u = 0.5*(velocity_d[Y*nsites_cd+index0] + velocity_d[Y*nsites_cd+index1]);


	  if (u < 0.0) {
	    fluxy_d[nop_cd*index0 + n] = u*phi_site_d[nsites_cd*n+index1];
	  }
	  else {
	    fluxy_d[nop_cd*index0 + n] = u*phi0;
	  }


	  /* z direction */


      index1=get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);
      u = 0.5*(velocity_d[Z*nsites_cd+index0] + velocity_d[Z*nsites_cd+index1]);

      if (u < 0.0) {
	fluxz_d[nop_cd*index0 + n] = u*phi_site_d[nsites_cd*n+index1];
      }
      else {
	fluxz_d[nop_cd*index0 + n] = u*phi0;
      }



      }
    }

  return;
}


__global__ void advection_bcs_no_normal_flux_gpu_d(const int nop,
					   char *site_map_status_d,
					   double *fluxe_d,
					   double *fluxw_d,
					   double *fluxy_d,
					   double *fluxz_d
					    ) {

  int ia, icm1, icp1;
  int index, index1, index0;
  int ii, jj, kk, n;
  double u, phi0;
  //int threadIndex,ii, jj, kk;
  double mask, maske, maskw, masky, maskz;


 /* CUDA thread index */
  //threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  // if (threadIndex < N_cd[X]*N_cd[Y]*N_cd[Z])

  if (ii < N_cd[X] && jj < (N_cd[Y]+1) && kk < (N_cd[Z]+1) )
    {


      /* calculate index from CUDA thread index */


      /* mask  = (site_map_get_status_index(index)  == FLUID); */
      /* maske = (site_map_get_status(ic+1, jc, kc) == FLUID); */
      /* maskw = (site_map_get_status(ic-1, jc, kc) == FLUID); */
      /* masky = (site_map_get_status(ic, jc+1, kc) == FLUID); */
      /* maskz = (site_map_get_status(ic, jc, kc+1) == FLUID); */
      
      
      index = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);      
      mask  = (site_map_status_d[index]  == FLUID); 

      index0 = get_linear_index_gpu_d(ii+nhalo_cd+1,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);      
      maske  = (site_map_status_d[index0]  == FLUID); 

      index0 = get_linear_index_gpu_d(ii+nhalo_cd-1,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);      
      maskw  = (site_map_status_d[index0]  == FLUID); 

      index0 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);      
      masky  = (site_map_status_d[index0]  == FLUID); 

      index0 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);      
      maskz  = (site_map_status_d[index0]  == FLUID); 




	/* for (n = 0;  n < nf; n++) { */
	/*   fluxw[nf*index + n] *= mask*maskw; */
	/*   fluxe[nf*index + n] *= mask*maske; */
	/*   fluxy[nf*index + n] *= mask*masky; */
	/*   fluxz[nf*index + n] *= mask*maskz; */
	/* } */


	for (n = 0;  n < nop; n++) { 
	   fluxw_d[nop*index + n] *= mask*maskw; 
	   fluxe_d[nop*index + n] *= mask*maske;
	   fluxy_d[nop*index + n] *= mask*masky; 
	   fluxz_d[nop*index + n] *= mask*maskz; 
	 } 
      
      


    }
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
