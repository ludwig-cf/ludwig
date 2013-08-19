/*****************************************************************************
 *
 *  phi_force_gpu.cu
 *
 *  GPU implementation of phi force and update functionality 
 *
 *  Alan Gray
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h> 

#define INCLUDING_FROM_GPU_SOURCE
#include "phi_force_gpu.h"
#include "phi_force_internal_gpu.h"
#include "comms_gpu.h"

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


dim3 nblocks, threadsperblock;

/* handles for CUDA streams (for ovelapping)*/
static cudaStream_t streamBULK, streamX, streamY, streamZ;


/*  phi_force_calculation_gpu - see CPU version in phi_force.c */

void phi_force_calculation_gpu(void) {

  int N[3],nhalo,Nall[3];
  nhalo = coords_nhalo();
  coords_nlocal(N); 
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  if (le_get_nplane_total() > 0 ) {
    printf("le_get_nplane_total() > 0 not yet supported in GPU mode. Exiting.\n");
    exit(1);
  }
  else if (wall_present()) {
    printf("wall_present()  not yet supported in GPU mode. Exiting.\n");
    exit(1);
  }
  else {


    put_phi_force_constants_on_gpu();

    expand_grad_phi_on_gpu();
    expand_phi_on_gpu();


  /* compute q2 and eq */
  threadsperblock.x=DEFAULT_TPB_Z;
  threadsperblock.y=DEFAULT_TPB_Y;
  threadsperblock.z=DEFAULT_TPB_X;

  nblocks.x=(Nall[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
  nblocks.y=(Nall[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
  nblocks.z=(Nall[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;

  cudaFuncSetCacheConfig(blue_phase_compute_q2_eq_all_gpu_d,cudaFuncCachePreferL1);      
  blue_phase_compute_q2_eq_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,tmpscal1_d,tmpscal2_d);

  cudaFuncSetCacheConfig(blue_phase_compute_h_all_gpu_d,cudaFuncCachePreferL1);      
  blue_phase_compute_h_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d, tmpscal1_d, tmpscal2_d);

  cudaFuncSetCacheConfig(blue_phase_compute_stress1_all_gpu_d,cudaFuncCachePreferL1);      
  blue_phase_compute_stress1_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,phi_site_full_d,grad_phi_site_d,grad_phi_site_full_d,delsq_phi_site_d,h_site_d, stress_site_d);


  /* decompose by a further factor of 9 for this part of stress calc for performance */

  threadsperblock.x=DEFAULT_TPB;
  threadsperblock.y=1;
  threadsperblock.z=1;

  /* can't do simple linear decomposition since gridDim.x gets too large. Need to use other grid dimentions and roll back in in kernel.*/

  nblocks.x=(9*Nall[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
  nblocks.y=Nall[Y];
  nblocks.z=Nall[X];

  cudaFuncSetCacheConfig(blue_phase_compute_stress2_all_gpu_d,cudaFuncCachePreferL1);      
  blue_phase_compute_stress2_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,grad_phi_site_full_d,stress_site_d);


  /* compute force */
  threadsperblock.x=DEFAULT_TPB_Z;
  threadsperblock.y=DEFAULT_TPB_Y;
  threadsperblock.z=DEFAULT_TPB_X;

  nblocks.x=(N[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
  nblocks.y=(N[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
  nblocks.z=(N[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;

      
  cudaFuncSetCacheConfig(phi_force_calculation_fluid_gpu_d,cudaFuncCachePreferL1);      
  phi_force_calculation_fluid_gpu_d<<<nblocks,threadsperblock>>>
    (le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,stress_site_d,force_d);
  
  cudaThreadSynchronize();
  checkCUDAError("phi_force_calculation_fluid_gpu_d");
  
  }
  return;
}

/*  phi_force_colloid_gpu - see CPU version in phi_force_coloid.c */
void phi_force_colloid_gpu(void) {

  int N[3],nhalo,Nall[3];
  nhalo = coords_nhalo();
  coords_nlocal(N); 
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
   
  if (colloids_q_anchoring_method() == ANCHORING_METHOD_ONE) {
    //phi_force_interpolation1();
    printf("ANCHORING_METHOD_ONE not yet supported in GPU mode. Exiting.\n");
    exit(1);
  }
  else {


    //phi_force_interpolation2();


    put_phi_force_constants_on_gpu();

    expand_grad_phi_on_gpu();
    expand_phi_on_gpu();


  /* compute q2 and eq */
  threadsperblock.x=DEFAULT_TPB_Z;
  threadsperblock.y=DEFAULT_TPB_Y;
  threadsperblock.z=DEFAULT_TPB_X;

  nblocks.x=(Nall[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
  nblocks.y=(Nall[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
  nblocks.z=(Nall[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;

  cudaFuncSetCacheConfig(blue_phase_compute_q2_eq_all_gpu_d,cudaFuncCachePreferL1);      
  blue_phase_compute_q2_eq_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,tmpscal1_d,tmpscal2_d);

  cudaFuncSetCacheConfig(blue_phase_compute_h_all_gpu_d,cudaFuncCachePreferL1);      
  blue_phase_compute_h_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d, tmpscal1_d, tmpscal2_d);

  cudaFuncSetCacheConfig(blue_phase_compute_stress1_all_gpu_d,cudaFuncCachePreferL1);      
  blue_phase_compute_stress1_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,phi_site_full_d,grad_phi_site_d,grad_phi_site_full_d,delsq_phi_site_d,h_site_d, stress_site_d);


  /* decompose by a further factor of 9 for this part of stress calc for performance */

  threadsperblock.x=DEFAULT_TPB;
  threadsperblock.y=1;
  threadsperblock.z=1;

  /* can't do simple linear decomposition since gridDim.x gets too large. Need to use other grid dimentions and roll back in in kernel.*/

  nblocks.x=(9*Nall[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
  nblocks.y=Nall[Y];
  nblocks.z=Nall[X];

  cudaFuncSetCacheConfig(blue_phase_compute_stress2_all_gpu_d,cudaFuncCachePreferL1);      
  blue_phase_compute_stress2_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,grad_phi_site_full_d,stress_site_d);


  /* compute force */
  threadsperblock.x=DEFAULT_TPB_Z;
  threadsperblock.y=DEFAULT_TPB_Y;
  threadsperblock.z=DEFAULT_TPB_X;

  nblocks.x=(N[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
  nblocks.y=(N[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
  nblocks.z=(N[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;

  cudaFuncSetCacheConfig(phi_force_colloid_gpu_d,cudaFuncCachePreferL1);      
  phi_force_colloid_gpu_d<<<nblocks,threadsperblock>>>
    (le_index_real_to_buffer_d,site_map_status_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,stress_site_d,force_d,colloid_force_d);
      
  cudaThreadSynchronize();
  checkCUDAError("phi_force_colloid_gpu_d");

  }

  return;
}


/*   blue_phase_be_update_gpu - see CPU version in blue_phase_beris_edwards.c */
void blue_phase_be_update_gpu(int async=0) {

 
 int N[3],nhalo,Nall[3];
  nhalo = coords_nhalo();
  coords_nlocal(N); 
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;


  cudaFuncSetCacheConfig(blue_phase_compute_q2_eq_all_gpu_d,cudaFuncCachePreferL1);      
  cudaFuncSetCacheConfig(blue_phase_compute_h_all_gpu_d,cudaFuncCachePreferL1);      
  cudaFuncSetCacheConfig(blue_phase_be_update_gpu_d,cudaFuncCachePreferL1);


  put_phi_force_constants_on_gpu();  
  expand_phi_on_gpu();
  expand_grad_phi_on_gpu();

  /* compute q2 and eq */
  threadsperblock.x=DEFAULT_TPB_Z;
  threadsperblock.y=DEFAULT_TPB_Y;
  threadsperblock.z=DEFAULT_TPB_X;
  
  nblocks.x=(Nall[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
  nblocks.y=(Nall[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
  nblocks.z=(Nall[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;
  
  
  
  blue_phase_compute_q2_eq_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,tmpscal1_d,tmpscal2_d);
  
  blue_phase_compute_h_all_gpu_d<<<nblocks,threadsperblock>>>(phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d, tmpscal1_d, tmpscal2_d);


  /* copy phi_site to phi_site_tmp on accelerator */
  double *tmpptr=phi_site_temp_d; phi_site_temp_d=phi_site_d; phi_site_d=tmpptr;


  /* need to make the edges 1D? */


  if (async==1){

    
    streamX=getXstream();streamY=getYstream();streamZ=getZstream();streamBULK=getBULKstream();
    
    
    
    /* X edges */
    threadsperblock.x=DEFAULT_TPB_Z; threadsperblock.y=DEFAULT_TPB_Y; threadsperblock.z=nhalo;
    
    nblocks.x=(N[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
    nblocks.y=(N[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
    nblocks.z=1;
  
    
    blue_phase_be_update_edge_gpu_d<<<nblocks,threadsperblock,0,streamX>>>
      (le_index_real_to_buffer_d,phi_site_d,phi_site_temp_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d,BE_UPDATE,X);
    
    /* Y edges */
    threadsperblock.x=DEFAULT_TPB_Z;
    threadsperblock.y=nhalo;
    threadsperblock.z=DEFAULT_TPB_X;;
    
    nblocks.x=(N[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
    nblocks.y=1;
    nblocks.z=(N[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;;
    
    cudaFuncSetCacheConfig(blue_phase_be_update_gpu_d,cudaFuncCachePreferL1);
    blue_phase_be_update_edge_gpu_d<<<nblocks,threadsperblock,0,streamY>>>
      (le_index_real_to_buffer_d,phi_site_d,phi_site_temp_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d,BE_UPDATE,Y);
    
    /* Z edges */
    threadsperblock.x=nhalo;
    threadsperblock.y=DEFAULT_TPB_Y;
    threadsperblock.z=DEFAULT_TPB_X;;
    
    nblocks.x=1;
    nblocks.y=(N[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
    nblocks.z=(N[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;
    
    blue_phase_be_update_edge_gpu_d<<<nblocks,threadsperblock,0,streamZ>>>
      (le_index_real_to_buffer_d,phi_site_d,phi_site_temp_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d,BE_UPDATE,Z);
    
    
    
    threadsperblock.x=DEFAULT_TPB_Z;
    threadsperblock.y=DEFAULT_TPB_Y;
    threadsperblock.z=DEFAULT_TPB_X;
    
    nblocks.x=(N[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
    nblocks.y=(N[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
    nblocks.z=(N[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;
    
    
    cudaFuncSetCacheConfig(blue_phase_be_update_gpu_d,cudaFuncCachePreferL1);
    blue_phase_be_update_gpu_d<<<nblocks,threadsperblock,0,streamBULK>>>
      (le_index_real_to_buffer_d,phi_site_d,phi_site_temp_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d,BE_UPDATE,BULK);
    
  }
  else{
    
    threadsperblock.x=DEFAULT_TPB_Z;
    threadsperblock.y=DEFAULT_TPB_Y;
    threadsperblock.z=DEFAULT_TPB_X;
    
    nblocks.x=(N[Z]+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
    nblocks.y=(N[Y]+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
    nblocks.z=(N[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;
    
    
    cudaFuncSetCacheConfig(blue_phase_be_update_gpu_d,cudaFuncCachePreferL1);
    blue_phase_be_update_gpu_d<<<nblocks,threadsperblock>>>
      (le_index_real_to_buffer_d,phi_site_d,phi_site_temp_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d,BE_UPDATE,ALL);
    
    cudaThreadSynchronize();
  }


      
  cudaThreadSynchronize();
  checkCUDAError("blue_phase_be_update_gpu_d");


  return;
}

/* advection_upwind_gpu - see CPU version in advection.c */
void advection_upwind_gpu(void) {

  int N[3];
  coords_nlocal(N); 
  

  put_phi_force_constants_on_gpu();

  // cudaFuncSetCacheConfig(advection_upwind_gpu_d,cudaFuncCachePreferL1);

  threadsperblock.x=DEFAULT_TPB_Z;
  threadsperblock.y=DEFAULT_TPB_Y;
  threadsperblock.z=DEFAULT_TPB_X;

 /* two fastest moving dimensions have cover a single lower width-one halo here */

  nblocks.x=(N[Z]+1+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z;
  nblocks.y=(N[Y]+1+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y;
  nblocks.z=(N[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X;

  cudaFuncSetCacheConfig(advection_upwind_gpu_d,cudaFuncCachePreferL1);
  advection_upwind_gpu_d<<<nblocks,threadsperblock>>>
    (le_index_real_to_buffer_d,phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,force_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d);
      
  cudaThreadSynchronize();
  checkCUDAError("advection_upwind_gpu_d");


  return;
}


/* advection_bcs_no_normal_flux_gpu - see CPU version in advection_bcs.c */
void advection_bcs_no_normal_flux_gpu(void){

  int N[3];
  
  coords_nlocal(N); 

  put_phi_force_constants_on_gpu();

  int nop = phi_nop();
  
  /* two fastest moving dimensions have cover a single lower width-one halo here */
  dim3 nblocks(((N[Z]+1)+DEFAULT_TPB_Z-1)/DEFAULT_TPB_Z,((N[Y]+1)+DEFAULT_TPB_Y-1)/DEFAULT_TPB_Y,(N[X]+DEFAULT_TPB_X-1)/DEFAULT_TPB_X);
  dim3 threadsperblock(DEFAULT_TPB_Z,DEFAULT_TPB_Y,DEFAULT_TPB_X);
  
  cudaFuncSetCacheConfig(advection_bcs_no_normal_flux_gpu_d,cudaFuncCachePreferL1);
  advection_bcs_no_normal_flux_gpu_d<<<nblocks,threadsperblock>>>
    (nop, site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d);
  
  cudaThreadSynchronize();
  checkCUDAError("advection_bcs_no_normal_flux_gpu");
  
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

__device__ double blue_phase_compute_fed_gpu_d(double q[3][3], double dq[3][3][3], const double* __restrict__ grad_phi_site_full_d, int index){

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

__device__ void blue_phase_compute_stress1_gpu_d(double q[3][3], double dq[3][3][3], double sth[3][3], const double* __restrict__ grad_phi_site_full_d, const double* __restrict__ h_site_d, int index){
  int ia, ib, ic;

  double tmpdbl,tmpdbl2;

  double h[3][3];

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib]=h_site_d[3*nsites_cd*ia+nsites_cd*ib+index];
    }
  }

  
  /* We have ignored the rho T term at the moment, assumed to be zero
   * (in particular, it has no divergence if rho = const). */

  tmpdbl = 0.0 - blue_phase_compute_fed_gpu_d(q, dq,grad_phi_site_full_d, index);

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
      // now done in second part of stress calc
      //sth[ia][ib] = -sth[ia][ib]; 
    }
  }






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

__global__ void phi_force_calculation_fluid_gpu_d(const int* __restrict__ le_index_real_to_buffer_d,
						  const double* __restrict__ phi_site_d,
						  const double* __restrict__ phi_site_full_d,
						  const double* __restrict__ grad_phi_site_d,
						  const double* __restrict__ delsq_phi_site_d,
						  const double* __restrict__ h_site_d,
						  const double* __restrict__ stress_site_d,
						  double* __restrict__ force_d){

  int ia, ib, icm1, icp1;
  int index, index1;
  double pth0[3][3];
  double pth1[3][3];
  double force[3];
  int ii, jj, kk;

  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  if (ii < N_cd[X] && jj < N_cd[Y] && kk < N_cd[Z] )
    {

      index = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);      
      icm1=le_index_real_to_buffer_d[ii+nhalo_cd];
      icp1=le_index_real_to_buffer_d[Nall_cd[X]+ii+nhalo_cd];      
      

	/* Compute pth at current point */
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth0[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index];
      	}
      }

	/* Compute differences */
	index1 = get_linear_index_gpu_d(icp1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }
	for (ia = 0; ia < 3; ia++) {
	  force[ia] = -0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	index1 = get_linear_index_gpu_d(icm1,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][X] + pth0[ia][X]);
	}

	
	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd+1,kk+nhalo_cd,Nall_cd);
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Y] + pth0[ia][Y]);
	}
	

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd+1,Nall_cd);
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }
	for (ia = 0; ia < 3; ia++) {
	  force[ia] -= 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	index1 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }
	for (ia = 0; ia < 3; ia++) {
	  force[ia] += 0.5*(pth1[ia][Z] + pth0[ia][Z]);
	}

	/* Store the force on lattice */
	for (ia=0;ia<3;ia++)
	  force_d[ia*nsites_cd+index]+=force[ia];

    }


  return;
}


__global__ void phi_force_colloid_gpu_d(const int * __restrict__ le_index_real_to_buffer_d,
					const char * __restrict__ site_map_status_d,
					const double* __restrict__ phi_site_d,
					const double* __restrict__ phi_site_full_d,
					const double* __restrict__ grad_phi_site_d,
					const double* __restrict__ delsq_phi_site_d,
					const double* __restrict__ h_site_d,
					const double* __restrict__ stress_site_d,
					double* __restrict__ force_d,
					double* __restrict__ colloid_force_d) {

  int ia, ib;
  int index, index1;
  double pth0[3][3];
  double pth1[3][3];
  double force[3];
  int ii, jj, kk;

  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */

  if (ii < N_cd[X] && jj < N_cd[Y] && kk < N_cd[Z] )
    {


      /* calculate index from CUDA thread index */
      index = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);      

      if (site_map_status_d[index] != COLLOID){
	/* If this is solid, then there's no contribution here. */
      
	/* Compute pth at current point */
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth0[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index];
      	}
      }

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
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }


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
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }


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
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }


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
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }


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
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }


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
      for (ia = 0; ia < 3; ia++) {
      	for (ib = 0; ib < 3; ib++) {
      	  pth1[ia][ib]=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index1];
      	}
      }


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

__device__ void blue_phase_be_update_site_gpu_d(const int * __restrict__ le_index_real_to_buffer_d,
					   double* __restrict__ phi_site_d,
					   const double* __restrict__ phi_site_temp_d,
					   const double* __restrict__ grad_phi_site_d,
					   const double* __restrict__ delsq_phi_site_d,
					   const double* __restrict__ h_site_d,
					   const double* __restrict__ velocity_d,
					   const char* __restrict__ site_map_status_d,
					   const double* __restrict__ fluxe_d,
					   const double* __restrict__ fluxw_d,
					   const double* __restrict__ fluxy_d,
						const double* __restrict__ fluxz_d,
						const int ii,const int jj,const int kk 
					   ){

  int icm1, icp1;
  int index, indexm1, indexp1;

  /* calculate index from CUDA thread index */
  
  index = get_linear_index_gpu_d(ii,jj,kk,Nall_cd);      
  
  icm1=le_index_real_to_buffer_d[ii];
  icp1=le_index_real_to_buffer_d[Nall_cd[X]+ii];      
  
  indexm1 = get_linear_index_gpu_d(icm1,jj,kk,Nall_cd);
  indexp1 = get_linear_index_gpu_d(icp1,jj,kk,Nall_cd);
  
  
  
  int ia, ib, id;
  
  double q[3][3];
  double d[3][3];
  double s[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  double w[3][3];
  double omega[3][3];
  double trace_qw;

  /* load phi */

  q[X][X] = phi_site_temp_d[nsites_cd*XX+index];
  q[X][Y] = phi_site_temp_d[nsites_cd*XY+index];
  q[X][Z] = phi_site_temp_d[nsites_cd*XZ+index];
  q[Y][X] = q[X][Y];
  q[Y][Y] = phi_site_temp_d[nsites_cd*YY+index];
  q[Y][Z] = phi_site_temp_d[nsites_cd*YZ+index];
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



   if (site_map_status_d[index] != FLUID) {
     
     q[X][X] += dt_solid_cd*Gamma_cd*h_site_d[3*nsites_cd*X+nsites_cd*X+index];
     q[X][Y] += dt_solid_cd*Gamma_cd*h_site_d[3*nsites_cd*X+nsites_cd*Y+index];
     q[X][Z] += dt_solid_cd*Gamma_cd*h_site_d[3*nsites_cd*X+nsites_cd*Z+index];
     q[Y][Y] += dt_solid_cd*Gamma_cd*h_site_d[3*nsites_cd*Y+nsites_cd*Y+index];
     q[Y][Z] += dt_solid_cd*Gamma_cd*h_site_d[3*nsites_cd*Y+nsites_cd*Z+index];
     
   }
   else {
     

	  /* Velocity gradient tensor, symmetric and antisymmetric parts */

       w[X][X] = 0.5*(velocity_d[X*nsites_cd+indexp1] - velocity_d[X*nsites_cd+indexm1]);
       w[Y][X] = 0.5*(velocity_d[Y*nsites_cd+indexp1] - velocity_d[Y*nsites_cd+indexm1]);
       w[Z][X] = 0.5*(velocity_d[Z*nsites_cd+indexp1] - velocity_d[Z*nsites_cd+indexm1]);
       
       indexm1 = get_linear_index_gpu_d(ii,jj-1,kk,Nall_cd);
       indexp1 = get_linear_index_gpu_d(ii,jj+1,kk,Nall_cd);

       w[X][Y] = 0.5*(velocity_d[X*nsites_cd+indexp1] - velocity_d[X*nsites_cd+indexm1]);
       w[Y][Y] = 0.5*(velocity_d[Y*nsites_cd+indexp1] - velocity_d[Y*nsites_cd+indexm1]);
       w[Z][Y] = 0.5*(velocity_d[Z*nsites_cd+indexp1] - velocity_d[Z*nsites_cd+indexm1]);

       indexm1 = get_linear_index_gpu_d(ii,jj,kk-1,Nall_cd);
       indexp1 = get_linear_index_gpu_d(ii,jj,kk+1,Nall_cd);

       w[X][Z] = 0.5*(velocity_d[X*nsites_cd+indexp1] - velocity_d[X*nsites_cd+indexm1]);
       w[Y][Z] = 0.5*(velocity_d[Y*nsites_cd+indexp1] - velocity_d[Y*nsites_cd+indexm1]);
       w[Z][Z] = 0.5*(velocity_d[Z*nsites_cd+indexp1] - velocity_d[Z*nsites_cd+indexm1]);
       

       /* Tracelessness */

       double tr = r3_cd*(w[X][X] + w[Y][Y] + w[Z][Z]);
       w[X][X] -= tr;
       w[Y][Y] -= tr;
       w[Z][Z] -= tr;



     //end  hydrodynamics_velocity_gradient_tensor(ic, jc, kc, w);
	  trace_qw = 0.0;

	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      trace_qw += q[ia][ib]*w[ib][ia];
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

	  int indexj, indexk;
	  indexj = get_linear_index_gpu_d(ii,jj-1,kk,Nall_cd);      
	  indexk = get_linear_index_gpu_d(ii,jj,kk-1,Nall_cd);      

	  q[X][X] += dt_cd*(s[X][X] + Gamma_cd*(h_site_d[3*nsites_cd*X+nsites_cd*X+index])
	  		 - fluxe_d[XX*nsites_cd+index] + fluxw_d[XX*nsites_cd+index]
	  		 - fluxy_d[XX*nsites_cd+index] + fluxy_d[XX*nsites_cd+indexj]
	  		 - fluxz_d[XX*nsites_cd+index] + fluxz_d[XX*nsites_cd+indexk]);

	  q[X][Y] += dt_cd*(s[X][Y] + Gamma_cd*(h_site_d[3*nsites_cd*X+nsites_cd*Y+index])
	  		 - fluxe_d[XY*nsites_cd+index] + fluxw_d[XY*nsites_cd+index]
	  		 - fluxy_d[XY*nsites_cd+index] + fluxy_d[XY*nsites_cd+indexj]
	  		 - fluxz_d[XY*nsites_cd+index] + fluxz_d[XY*nsites_cd+indexk]);

	  q[X][Z] += dt_cd*(s[X][Z] + Gamma_cd*(h_site_d[3*nsites_cd*X+nsites_cd*Z+index])
	  		 - fluxe_d[XZ*nsites_cd+index] + fluxw_d[XZ*nsites_cd+index]
	  		 - fluxy_d[XZ*nsites_cd+index] + fluxy_d[XZ*nsites_cd+indexj]
	  		 - fluxz_d[XZ*nsites_cd+index] + fluxz_d[XZ*nsites_cd+indexk]);

	  q[Y][Y] += dt_cd*(s[Y][Y] + Gamma_cd*(h_site_d[3*nsites_cd*Y+nsites_cd*Y+index])
	  		 - fluxe_d[YY*nsites_cd+index] + fluxw_d[YY*nsites_cd+index]
	  		 - fluxy_d[YY*nsites_cd+index] + fluxy_d[YY*nsites_cd+indexj]
	  		 - fluxz_d[YY*nsites_cd+index] + fluxz_d[YY*nsites_cd+indexk]);

	  q[Y][Z] += dt_cd*(s[Y][Z] + Gamma_cd*(h_site_d[3*nsites_cd*Y+nsites_cd*Z+index])
	  		 - fluxe_d[YZ*nsites_cd+index] + fluxw_d[YZ*nsites_cd+index]
	  		 - fluxy_d[YZ*nsites_cd+index] + fluxy_d[YZ*nsites_cd+indexj]
	  		 - fluxz_d[YZ*nsites_cd+index] + fluxz_d[YZ*nsites_cd+indexk]);
	

	   }
	 phi_site_d[nsites_cd*XX+index] = q[X][X];
	 phi_site_d[nsites_cd*XY+index] = q[X][Y];
	 phi_site_d[nsites_cd*XZ+index] = q[X][Z];
	 phi_site_d[nsites_cd*YY+index] = q[Y][Y];
	 phi_site_d[nsites_cd*YZ+index] = q[Y][Z];


    


  return;
}


__global__ void blue_phase_be_update_gpu_d(const int * __restrict__ le_index_real_to_buffer_d,
					   double* __restrict__ phi_site_d,
					   const double* __restrict__ phi_site_temp_d,
					   const double* __restrict__ grad_phi_site_d,
					   const double* __restrict__ delsq_phi_site_d,
					   const double* __restrict__ h_site_d,
					   const double* __restrict__ velocity_d,
					   const char* __restrict__ site_map_status_d,
					   const double* __restrict__ fluxe_d,
					   const double* __restrict__ fluxw_d,
					   const double* __restrict__ fluxy_d,
					   const double* __restrict__ fluxz_d,
					   const int calcstep, const int latchunk
					   ){

  int icm1, icp1;
  int index, indexm1, indexp1;
  int ii, jj, kk;


  int edgeoffset;
  if (latchunk==BULK)
    edgeoffset=2*nhalo_cd;
  else
    edgeoffset=nhalo_cd;

 /* CUDA thread index */
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  if (ii < (Nall_cd[X]-2*edgeoffset) && jj < (Nall_cd[Y]-2*edgeoffset) && kk < (Nall_cd[X]-2*edgeoffset) )
    {

      if (calcstep == BE_UPDATE){
      blue_phase_be_update_site_gpu_d(le_index_real_to_buffer_d,phi_site_d,phi_site_temp_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d,ii+edgeoffset,jj+edgeoffset,kk+edgeoffset);
      }


    }


  return;
}

__global__ void blue_phase_be_update_edge_gpu_d(const int * __restrict__ le_index_real_to_buffer_d,
					   double* __restrict__ phi_site_d,
					   const double* __restrict__ phi_site_temp_d,
					   const double* __restrict__ grad_phi_site_d,
					   const double* __restrict__ delsq_phi_site_d,
					   const double* __restrict__ h_site_d,
					   const double* __restrict__ velocity_d,
					   const char* __restrict__ site_map_status_d,
					   const double* __restrict__ fluxe_d,
					   const double* __restrict__ fluxw_d,
					   const double* __restrict__ fluxy_d,
					   const double* __restrict__ fluxz_d,
					   const int calcstep, const int dirn
					   ){

  int icm1, icp1;
  int index, indexm1, indexp1;
  int ii, jj, kk;

  int Nedge[3];
  int edgeoffset; 

  if (dirn == X){
    Nedge[X]=nhalo_cd;
    Nedge[Y]=N_cd[Y];
    Nedge[Z]=N_cd[Z];
  }
  else if (dirn == Y){
    Nedge[X]=N_cd[X];
    Nedge[Y]=nhalo_cd;
    Nedge[Z]=N_cd[Z];
  }
  else if (dirn == Z){
    Nedge[X]=N_cd[X];
    Nedge[Y]=N_cd[Y];
    Nedge[Z]=nhalo_cd;
  }


 /* CUDA thread index */
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;

       int ii_,jj_,kk_;
 
 /* Avoid going beyond problem domain */
  if (ii < Nedge[X] && jj < Nedge[Y] && kk < Nedge[Z] )
    {

      
      ii_= ii+nhalo_cd;jj_=jj+nhalo_cd;kk_=kk+nhalo_cd;

      /* LOW EDGE */

      blue_phase_be_update_site_gpu_d(le_index_real_to_buffer_d,phi_site_d,phi_site_temp_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d,ii_,jj_,kk_);

        
      /* HIGH EDGE */
      if (dirn == X){
      	ii_=Nall_cd[X]-2*nhalo_cd+ii;jj_=jj+nhalo_cd;kk_=kk+nhalo_cd;
      }
      else if (dirn == Y){
      	ii_=ii+nhalo_cd;jj_=Nall_cd[Y]-2*nhalo_cd+jj;kk_=kk+nhalo_cd;
      }
      else if (dirn == Z){
      	ii_=ii+nhalo_cd;;jj_=jj+nhalo_cd;;kk_=Nall_cd[Z]-2*nhalo_cd+kk;
      }

      blue_phase_be_update_site_gpu_d(le_index_real_to_buffer_d,phi_site_d,phi_site_temp_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,velocity_d,site_map_status_d, fluxe_d, fluxw_d, fluxy_d, fluxz_d,ii_,jj_,kk_);


    }


  return;
}

__global__ void advection_upwind_gpu_d(const int * __restrict__ le_index_real_to_buffer_d,
				       const double* __restrict__ phi_site_d,
				       const double* __restrict__ phi_site_full_d,
				       const double* __restrict__ grad_phi_site_d,
				       const double* __restrict__ delsq_phi_site_d,
				       const double* __restrict__ force_d, 
				       const double* __restrict__ velocity_d,
				       const char* __restrict__site_map_status_d,
				       double* __restrict__ fluxe_d,
				       double* __restrict__ fluxw_d,
				       double* __restrict__ fluxy_d,
				       double* __restrict__ fluxz_d
				       ){

  int icm1, icp1;
  int index1, index0;
  int ii, jj, kk, n;
  double u, phi0;

  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  if (ii < N_cd[X] && jj < (N_cd[Y]+1) && kk < (N_cd[Z]+1) )
    {


      index0 = get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);      
      icm1=le_index_real_to_buffer_d[ii+nhalo_cd];
      icp1=le_index_real_to_buffer_d[Nall_cd[X]+ii+nhalo_cd];      


      
      for (n = 0; n < nop_cd; n++) {

	phi0 = phi_site_d[nsites_cd*n+index0];
	index1=get_linear_index_gpu_d(icm1,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);
	u = 0.5*(velocity_d[X*nsites_cd+index0] + velocity_d[X*nsites_cd+index1]);
	
	if (u > 0.0) {
	  fluxw_d[n*nsites_cd+index0] = u*phi_site_d[nsites_cd*n+index1];
	}
	else {
	  fluxw_d[n*nsites_cd+index0] = u*phi0;
	}

	  /* east face (ic and icp1) */

	index1=get_linear_index_gpu_d(icp1,jj+nhalo_cd-1,kk+nhalo_cd-1,Nall_cd);
	  u = 0.5*(velocity_d[X*nsites_cd+index0] + velocity_d[X*nsites_cd+index1]);

	  if (u < 0.0) {
	    fluxe_d[n*nsites_cd+index0] = u*phi_site_d[nsites_cd*n+index1];
	  }
	  else {
	    fluxe_d[n*nsites_cd+index0] = u*phi0;
	  }


	  /* y direction */

	index1=get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd-1,Nall_cd);
	  u = 0.5*(velocity_d[Y*nsites_cd+index0] + velocity_d[Y*nsites_cd+index1]);


	  if (u < 0.0) {
	    fluxy_d[n*nsites_cd+index0] = u*phi_site_d[nsites_cd*n+index1];
	  }
	  else {
	    fluxy_d[n*nsites_cd+index0] = u*phi0;
	  }


	  /* z direction */


      index1=get_linear_index_gpu_d(ii+nhalo_cd,jj+nhalo_cd-1,kk+nhalo_cd,Nall_cd);
      u = 0.5*(velocity_d[Z*nsites_cd+index0] + velocity_d[Z*nsites_cd+index1]);

      if (u < 0.0) {
	fluxz_d[n*nsites_cd+index0] = u*phi_site_d[nsites_cd*n+index1];
      }
      else {
	fluxz_d[n*nsites_cd+index0] = u*phi0;
      }



      }
    }

  return;
}

__global__ void advection_bcs_no_normal_flux_gpu_d(const int nop,
					   const char* __restrict__ site_map_status_d,
					   double* __restrict__ fluxe_d,
					   double* __restrict__ fluxw_d,
					   double* __restrict__ fluxy_d,
					   double* __restrict__ fluxz_d
						   ){

  int index,  index0;
  int ii, jj, kk, n;
  double mask, maske, maskw, masky, maskz;


  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
 /* Avoid going beyond problem domain */
  if (ii < N_cd[X] && jj < (N_cd[Y]+1) && kk < (N_cd[Z]+1) )
    {


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





	for (n = 0;  n < nop; n++) { 
	   fluxw_d[n*nsites_cd+index] *= mask*maskw; 
	   fluxe_d[n*nsites_cd+index] *= mask*maske;
	   fluxy_d[n*nsites_cd+index] *= mask*masky; 
	   fluxz_d[n*nsites_cd+index] *= mask*maskz; 
	 } 
      
      


    }
}

__device__ void blue_phase_compute_q2_eq_site_gpu_d( const double* __restrict__ phi_site_d,
						 const double* __restrict__ phi_site_full_d,
						 const double* __restrict__ grad_phi_site_d,
						 const double* __restrict__ delsq_phi_site_d,
						 const double* __restrict__ h_site_d,
						 double* __restrict__ q2_site_d,
						     double* __restrict__ eq_site_d,
						  const int ii, const int jj, const int kk){

  int ia, ib, ic;
  int index;
  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];
 
  if (ii < Nall_cd[X] && jj < Nall_cd[Y] && kk < Nall_cd[Z] )
    {


      /* calculate index from CUDA thread index */
      index = get_linear_index_gpu_d(ii,jj,kk,Nall_cd);
      
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
                  
  double q2;
  double eq;
  /* From the bulk terms in the free energy... */

  q2 = 0.0;
  eq = 0.0;
  
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      q2 += phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index]*phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index];
      
      for (ic = 0; ic < 3; ic++) {
	eq += e_cd[ia][ib][ic]*dq[ia][ib][ic];
      }
      
    }
  }

  q2_site_d[index]=q2;
  eq_site_d[index]=eq;
    }
  return;
}


__global__ void blue_phase_compute_q2_eq_all_gpu_d( const double* __restrict__ phi_site_d,
						 const double* __restrict__ phi_site_full_d,
						 const double* __restrict__ grad_phi_site_d,
						 const double* __restrict__ delsq_phi_site_d,
						 const double* __restrict__ h_site_d,
						 double* __restrict__ q2_site_d,
						    double* __restrict__ eq_site_d){

  int ia, ib, ic;
  int index;
  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  int ii, jj, kk;
 
 /* CUDA thread index */
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;


  blue_phase_compute_q2_eq_site_gpu_d(phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d,q2_site_d,eq_site_d,ii,jj,kk);

  return;
}







__device__ void blue_phase_compute_h_site_gpu_d(  const double* __restrict__ phi_site_d,
						 const double* __restrict__ phi_site_full_d,
						 const double* __restrict__ grad_phi_site_d,
						 const double* __restrict__ delsq_phi_site_d,
						 double* __restrict__ h_site_d,
						 const double* __restrict__ q2_site_d,
						  const double* __restrict__ eq_site_d,
						  const int ii, const int jj, const int kk
						 ){

  int ia, ib, ic, id;
  int index;
  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  double htmp;

 
  if (ii < Nall_cd[X] && jj < Nall_cd[Y] && kk < Nall_cd[Z] )
    {


      index = get_linear_index_gpu_d(ii,jj,kk,Nall_cd);
      
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
                  
  double sum, sum1;

  double q2=q2_site_d[index];
  double eq=eq_site_d[index];


  double cd1=-a0_cd*(1.0 - r3_cd*gamma_cd);
  double cd2=a0_cd*gamma_cd;
  double cd3=2.0*kappa1shift_cd*q0shift_cd;
  
  /* d_c Q_db written as d_c Q_bd etc */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      sum = 0.0;
      sum1 = 0.0;
      for (ic = 0; ic < 3; ic++) {

  	sum +=  phi_site_full_d[3*nsites_cd*ia+nsites_cd*ic+index]* phi_site_full_d[3*nsites_cd*ib+nsites_cd*ic+index];

	for (id = 0; id < 3; id++) {
	  sum1 +=
	    (e_cd[ia][ic][id]*dq[ic][ib][id] + e_cd[ib][ic][id]*dq[ic][ia][id]);
	}
      }

      htmp = cd1* phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index]
  	+ cd2*(sum - r3_cd*q2*d_cd[ia][ib]) - cd2*q2*phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index];

      htmp += kappa0shift_cd*dsq[ia][ib]
	- cd3*sum1 + 4.0*r3_cd*kappa1shift_cd*q0shift_cd*eq*d_cd[ia][ib]
	- 4.0*kappa1shift_cd*q0shift_cd*q0shift_cd*phi_site_full_d[3*nsites_cd*ia+nsites_cd*ib+index];

      htmp +=  epsilon_cd*(electric_cd[ia]*electric_cd[ib] - r3_cd*d_cd[ia][ib]*e2_cd);

       h_site_d[3*nsites_cd*ia+nsites_cd*ib+index]=htmp;

    }
  }
  
    }

  return;
}

__global__ void blue_phase_compute_h_all_gpu_d(  const double* __restrict__ phi_site_d,
						 const double* __restrict__ phi_site_full_d,
						 const double* __restrict__ grad_phi_site_d,
						 const double* __restrict__ delsq_phi_site_d,
						 double* __restrict__ h_site_d,
						 const double* __restrict__ q2_site_d,
						 const double* __restrict__ eq_site_d
						 ){

  int ia, ib, ic, id;
  int index;
  double q[3][3];
  double dq[3][3][3];
  double dsq[3][3];
  double htmp;

  int ii, jj, kk;
 
 /* CUDA thread index */
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;


  blue_phase_compute_h_site_gpu_d(phi_site_d,phi_site_full_d,grad_phi_site_d,delsq_phi_site_d,h_site_d, q2_site_d,eq_site_d,ii,jj,kk);


  return;
}


__global__ void blue_phase_compute_stress1_all_gpu_d(const  double* __restrict__ phi_site_d,
						 const double* __restrict__ phi_site_full_d,
						 const double* __restrict__ grad_phi_site_d,
						 const double* __restrict__ grad_phi_site_full_d,
						 const double* __restrict__ delsq_phi_site_d,
						 const     double* __restrict__ h_site_d,
						     double* __restrict__ stress_site_d){

  int ia, ib;
  int index;
  double q[3][3];
  double sth[3][3];
  double dq[3][3][3];
  double dsq[3][3];

  int ii, jj, kk;
 
  kk = blockIdx.x*blockDim.x+threadIdx.x;
  jj = blockIdx.y*blockDim.y+threadIdx.y;
  ii = blockIdx.z*blockDim.z+threadIdx.z;
 
  if (ii < Nall_cd[X] && jj < Nall_cd[Y] && kk < Nall_cd[Z] )
    {

      index = get_linear_index_gpu_d(ii,jj,kk,Nall_cd);
      
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
                  
      
      blue_phase_compute_stress1_gpu_d(q, dq, sth, grad_phi_site_full_d,h_site_d, index);

      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index]=sth[ia][ib];
	}
      }
    }

  return;
}

__global__ void blue_phase_compute_stress2_all_gpu_d(const  double* __restrict__ phi_site_d,
						 const double* __restrict__ grad_phi_site_full_d,
						     double* __restrict__ stress_site_d){

  int index;
  double q[3][3];
  
  double sth_tmp;
  

  //    int threadIndex = (blockIdx.x*blockDim.x+threadIdx.x);

    int threadIndex = blockIdx.z*gridDim.y*gridDim.x*blockDim.x
    + blockIdx.y*gridDim.x*blockDim.x
    + (blockIdx.x*blockDim.x+threadIdx.x);


  int i=threadIndex/(Nall_cd[Y]*Nall_cd[Z]*9);
  int j=(threadIndex-i*Nall_cd[Y]*Nall_cd[Z]*9)/(Nall_cd[Z]*9);
  int k=(threadIndex-i*Nall_cd[Y]*Nall_cd[Z]*9-j*Nall_cd[Z]*9)/9;
  int iw=threadIndex-i*Nall_cd[Y]*Nall_cd[Z]*9-j*Nall_cd[Z]*9-k*9;
  int ib=iw/3;
  int ia=iw-ib*3;

  
 
  if (i < Nall_cd[X] && j < Nall_cd[Y] && k < Nall_cd[Z] && ia<3 && ib <3)
    {

      index = get_linear_index_gpu_d(i,j,k,Nall_cd);
      
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
      
      

      sth_tmp=stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index];


      int ic, id, ie;
      
      double tmpdbl,tmpdbl2;
      
      /* Dot product term d_a Q_cd . dF/dQ_cd,b */
      
      tmpdbl=0.;
      for (ic = 0; ic < 3; ic++) {
      	for (id = 0; id < 3; id++) {
	  
      	  tmpdbl +=
      	    - kappa0shift_cd*grad_phi_site_full_d[ia*nsites_cd*9 + ib*nsites_cd*3+ ic*nsites_cd + index]*
      	    grad_phi_site_full_d[id*nsites_cd*9 + ic*nsites_cd*3+ id*nsites_cd + index]
      	    - kappa1shift_cd*grad_phi_site_full_d[ia*nsites_cd*9 + ic*nsites_cd*3+ id*nsites_cd + index]*
      	    grad_phi_site_full_d[ib*nsites_cd*9 + ic*nsites_cd*3+ id*nsites_cd + index]
      	    + kappa1shift_cd*grad_phi_site_full_d[ia*nsites_cd*9 + ic*nsites_cd*3+ id*nsites_cd + index]*
      	    grad_phi_site_full_d[ic*nsites_cd*9 + ib*nsites_cd*3+ id*nsites_cd + index];
	  
      	  tmpdbl2= -2.0*kappa1shift_cd*q0shift_cd
      	    *grad_phi_site_full_d[ia*nsites_cd*9 + ic*nsites_cd*3+ id*nsites_cd + index];
      	  for (ie = 0; ie < 3; ie++) {
      	    tmpdbl +=
      	      tmpdbl2*e_cd[ib][ic][ie]*q[id][ie];
      	  }



      	}
      }
      sth_tmp+=tmpdbl;

      sth_tmp=-sth_tmp;
      stress_site_d[3*nsites_cd*ia+nsites_cd*ib+index]=sth_tmp;
    }

  return;
}

void put_phi_force_constants_on_gpu(){


  int N[3],nhalo,Nall[3];
  
  nhalo = coords_nhalo();
  coords_nlocal(N); 


  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  
  int nsites=Nall[X]*Nall[Y]*Nall[Z];
  int nop = phi_nop();


  double redshift_ = blue_phase_redshift(); 
  double rredshift_ = blue_phase_rredshift(); 
  double q0_=blue_phase_q0();
  q0_ = q0_*rredshift_;

  double a0_=blue_phase_a0();
  double kappa0_=blue_phase_kappa0();
  double kappa1_=blue_phase_kappa1();
  kappa0_ = kappa0_*redshift_*redshift_;
  kappa1_ = kappa1_*redshift_*redshift_;

  double xi_=blue_phase_get_xi();
  double zeta_=blue_phase_get_zeta();
  double gamma_=blue_phase_gamma();
  double epsilon_=blue_phase_get_dielectric_anisotropy();
  double Gamma_=blue_phase_be_get_rotational_diffusion();

  double electric_[3];
  blue_phase_get_electric_field(electric_);  
  int ia;
  double e2=0;
  for (ia = 0; ia < 3; ia++) 
    e2 += electric_[ia]*electric_[ia];   /* Electric field term */

  /* For first anchoring method (only) have evolution at solid sites. */
  const double dt = 1.0;
  double dt_solid;
  dt_solid = 0;
  if (colloids_q_anchoring_method() == ANCHORING_METHOD_ONE) dt_solid = dt;


  double cd1_=-a0_*(1.0 - r3_*gamma_);
  double cd2_=a0_*gamma_;
  double cd3_=2.0*kappa1_*q0_;
  double cd4_=r3_*kappa1_*q0_;
  double cd5_=4.0*kappa1_*q0_*q0_;
  double cd6_=r3_*e2;

  /* copy to constant memory on device */
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
  cudaMemcpyToSymbol(e2_cd, &e2, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(d_cd, d_, 3*3*sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(e_cd, e_, 3*3*3*sizeof(double), 0, cudaMemcpyHostToDevice);
  		
  cudaMemcpyToSymbol(cd1, &cd1_, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(cd2, &cd2_, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(cd3, &cd3_, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(cd4, &cd4_, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(cd5, &cd5_, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(cd6, &cd6_, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nop_cd, &nop, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(dt_solid_cd, &dt_solid, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(dt_cd, &dt, sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(Gamma_cd, &Gamma_, sizeof(double), 0, cudaMemcpyHostToDevice); 

 
  checkCUDAError("phi_force cudaMemcpyToSymbol");


}


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N_d[3])
{
  
  int yfac = N_d[Z];
  int xfac = N_d[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}
