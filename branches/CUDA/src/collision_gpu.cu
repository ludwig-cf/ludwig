 /*****************************************************************************
 *
 *  collision_gpu.c
 *
 *  GPU implementation of collision functionality
 * 
 *  Alan Gray
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>


/* below define needed to stop repeated declaration of distribution_ndist */
#define INCLUDING_FROM_GPU_SOURCE 

#include "model.h"
#include "site_map.h"
#include "comms_gpu.h"
#include "collision_gpu.h"
#include "collision_internal_gpu.h"
#include "collision.h"
#include "fluctuations.h"

//#include "free_energy.h"
//#include "phi_cahn_hilliard.h"

#include "control.h"
#include "propagation_ode.h"

static int nrelax_ = RELAXATION_M10;     /* [RELAXATION_M10|TRT|BGK] */
                                         /* Default is M10 */
static int isothermal_fluctuations_ = 0; /* Flag for noise. */

static double rtau_shear;       /* Inverse relaxation time for shear modes */
static double rtau_bulk;        /* Inverse relaxation time for bulk modes */
static double var_shear;        /* Variance for shear mode fluctuations */
static double var_bulk;         /* Variance for bulk mode fluctuations */
static double rtau_[NVEL];      /* Inverse relaxation times */
static double noise_var[NVEL];  /* Noise variances */

static double rtau2;

extern const double d_[3][3];

/* handles for CUDA streams (for ovelapping)*/
static cudaStream_t streamCOLL, streamX, streamY, streamZ;




void collide_gpu(int async=0) {

  int ndist,nhalo;
  double mobility;
  int N[3];

  int Nall[3];

  ndist = distribution_ndist();
  nhalo = coords_nhalo();
  coords_nlocal(N); 

  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  collision_relaxation_times_set_gpu();

  mobility = phi_cahn_hilliard_mobility();
  rtau2 = 2.0 / (1.0 + 2.0*mobility);


  /* copy f to ftmp on accelerator */
  //copy_f_to_ftmp_on_gpu();
  
  double *tmpptr=ftmp_d;
  ftmp_d=f_d;
  f_d=tmpptr;


/* copy constants to accelerator (constant on-chip read-only memory) */
  copy_constants_to_gpu();

  cudaStreamCreate(&streamCOLL);
  
  int colltype;

  int nblocks;
  /* set up CUDA grid */
  /* 1D decomposition - use x grid and block dimension only */ 
  //int nblocks=(N[X]*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;

  if ((ndist == 1 || is_propagation_ode() == 1 ) && nrelax_ == RELAXATION_M10)

    {

      colltype=MULTIRELAXATION;
    }
      
  else if  (ndist == 2 && is_propagation_ode() == 0) 
    { 

      colltype=BINARY;
    }

  //if ((ndist == 1 || is_propagation_ode() == 1 ) && nrelax_ == RELAXATION_BGK)
  //  {
  //    printf("Error, KGK collision not supported yet in GPU version\n");
  //    exit(1);

  //   }


  else
    {
      printf("Error, the requested collision is not supported yet in GPU version\n");
      exit(1);

    }




  streamX=getXstream();
  streamY=getYstream();
  streamZ=getZstream();
      


  /* Bulk */
  nblocks=((N[X]-2*nhalo)*(N[Y]-2*nhalo)*(N[Z]-2*nhalo)+DEFAULT_TPB-1)/DEFAULT_TPB;

  collision_lb_gpu_d<<<nblocks,DEFAULT_TPB,0,streamCOLL>>>(ndist, nhalo, N_d, 					      force_global_d,
  					      f_d, ftmp_d,
  					      site_map_status_d,
  					       phi_site_d,
  					       grad_phi_site_d,
  					       delsq_phi_site_d,
  							 force_d,
							   velocity_d,colltype, BULK);


 /* X edges */
 nblocks=(nhalo*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
 collision_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamX>>>(nhalo,
 						       N_d,force_global_d,
 					      f_d, ftmp_d,
 					      site_map_status_d,
 					       phi_site_d,
 					       grad_phi_site_d,
 					       delsq_phi_site_d,
 							 force_d,
							 velocity_d,colltype, X);

 /* Y edges */
  nblocks=((N[X]-2*nhalo)*nhalo*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
  collision_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamY>>>(nhalo,
 						       N_d,force_global_d,
 					      f_d, ftmp_d,
 					      site_map_status_d,
 					       phi_site_d,
 					       grad_phi_site_d,
 					       delsq_phi_site_d,
 							 force_d,
							  velocity_d,colltype, Y);

 /* Z edges */
  nblocks=((N[X]-2*nhalo)*(N[Y]-2*nhalo)*nhalo+DEFAULT_TPB-1)/DEFAULT_TPB;
  collision_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamZ>>>(nhalo,
 						       N_d,force_global_d,
 					      f_d, ftmp_d,
 					      site_map_status_d,
 					       phi_site_d,
 					       grad_phi_site_d,
 					       delsq_phi_site_d,
 							 force_d,
							  velocity_d,colltype, Z);


  //KEEP THESE SYNC POINTS IN JUST NOW UNTL SURE RACE-FREE RE. CORNERS
  cudaStreamSynchronize(streamX);
  cudaStreamSynchronize(streamY);
  cudaStreamSynchronize(streamZ);

  //if (async==1)
  //  cudaStreamSynchronize(streamCOLL);



  if (async==0){
    //cudaStreamSynchronize(streamX);
    //cudaStreamSynchronize(streamY);
    //cudaStreamSynchronize(streamZ);
  cudaStreamSynchronize(streamCOLL);

  cudaStreamDestroy(streamCOLL);
  }

  return;
}

void collide_wait_gpu()
{
   cudaStreamSynchronize(streamCOLL);
   cudaStreamDestroy(streamCOLL);

   return;
}

__device__ void collision_multirelaxation_site_gpu_d(
					      const double* __restrict__ force_global_d, 
					      double* __restrict__ f_d, 
					      const double* __restrict__ ftmp_d, 

					      const char* __restrict__ site_map_status_d, 
					      const double* __restrict__ force_d, 
    			       		      double* __restrict__ velocity_d, const int nsite, const int index
					      )
{

  int       p, m;                    /* velocity index */
  int       ia, ib;                  /* indices ("alphabeta") */

  double    mode[NVEL];              /* Modes; hydrodynamic + ghost */
  double    rho, rrho;               /* Density, reciprocal density */
  double    u[3];                    /* Velocity */
  double    s[3][3];                 /* Stress */
  double    seq[3][3];               /* Equilibrium stress */
  double    shat[3][3];              /* random stress */
  double    ghat[NVEL];              /* noise for ghosts */
  double    rdim;                    /* 1 / dimension */

  double    force[3];                /* External force */
  double    tr_s, tr_seq;

  double    force_local[3];

  /* int threadIndex, nsite, Nall[3], ii, jj, kk, xfac, yfac; */

  
	  
	  
	  /* Compute all the modes */
	  
	  for (m = 0; m < NVEL; m++) {
	   double dtmp = 0.;
	  for (p = 0; p < NVEL; p++) {
	    dtmp += ftmp_d[nsite*p + index]*ma_cd[m][p];
	  }
	  mode[m] = dtmp;
	  }
	  

	  /* For convenience, write out the physical modes, that is,
	   * rho, NDIM components of velocity, independent components
	   * of stress (upper triangle), and lower triangle. */
	  
	  rho = mode[0];
	  for (ia = 0; ia < NDIM; ia++) {
	    u[ia] = mode[1 + ia];
	  }
	  
	  
	  m = 0;
	  for (ia = 0; ia < NDIM; ia++) {
	    for (ib = ia; ib < NDIM; ib++) {
	      s[ia][ib] = mode[1 + NDIM + m++];
	    }
	  }
	  
	  for (ia = 1; ia < NDIM; ia++) {
	    for (ib = 0; ib < ia; ib++) {
	      s[ia][ib] = s[ib][ia];
	    }
	  }
	  
	  
	  
	  /* Compute the local velocity, taking account of any body force */
	  
	  rrho = 1.0/rho;
	  /* hydrodynamics_get_force_local(index, force_local); */
	  for (ia = 0; ia < 3; ia++) {
	    force_local[ia] = force_d[ia*nsite+index];
	  }
	  
	  for (ia = 0; ia < NDIM; ia++) {
	    force[ia] = (force_global_d[ia] + force_local[ia]);
	    u[ia] = rrho*(u[ia] + 0.5*force[ia]);
	  }
	  
	  /* hydrodynamics_set_velocity(index, u); */
	  for (ia = 0; ia < 3; ia++) {
	    velocity_d[ia*nsite+index] = u[ia];
	  }
	  
	  /* Relax stress with different shear and bulk viscosity */
	  
	  tr_s   = 0.0;
	  tr_seq = 0.0;
	  
	  for (ia = 0; ia < NDIM; ia++) {
	    /* Set equilibrium stress */
	    for (ib = 0; ib < NDIM; ib++) {
	      seq[ia][ib] = rho*u[ia]*u[ib];
	    }
	    /* Compute trace */
	    tr_s   += s[ia][ia];
	    tr_seq += seq[ia][ia];
	  }
	  
	  /* Form traceless parts */
	  for (ia = 0; ia < NDIM; ia++) {
	    s[ia][ia]   -= rdim*tr_s;
	    seq[ia][ia] -= rdim*tr_seq;
	  }
	  
	  /* Relax each mode */
	  tr_s = tr_s - rtau_bulk_d*(tr_s - tr_seq);
	  
	  for (ia = 0; ia < NDIM; ia++) {
	    for (ib = 0; ib < NDIM; ib++) {
	      s[ia][ib] -= rtau_shear_d*(s[ia][ib] - seq[ia][ib]);
	      s[ia][ib] += d_cd[ia][ib]*rdim*tr_s;
	      
	      /* Correction from body force (assumes equal relaxation times) */
	      
	      s[ia][ib] += (2.0-rtau_shear_d)*(u[ia]*force[ib] + force[ia]*u[ib]);
	    }
	  }
	  
	  //if (isothermal_fluctuations_) 	    collision_fluctuations(index, shat, ghat);
	  
	  /* Now reset the hydrodynamic modes to post-collision values:
	   * rho is unchanged, velocity unchanged if no force,
	   * independent components of stress, and ghosts. */
	  
	  for (ia = 0; ia < NDIM; ia++) {
	    mode[1 + ia] += force[ia];
	  }

	  
	  m = 0;
	  for (ia = 0; ia < NDIM; ia++) {
	    for (ib = ia; ib < NDIM; ib++) {
	      mode[1 + NDIM + m++] = s[ia][ib] + shat[ia][ib];
	    }
	  }
	  
	  /* Ghost modes are relaxed toward zero equilibrium. */
	  
	  for (m = NHYDRO; m < NVEL; m++) {
	    mode[m] = mode[m] - rtau_d[m]*(mode[m] - 0.0) + ghat[m];
	  }
	  
	  /* Project post-collision modes back onto the distribution */
	  
	  for (p = 0; p < NVEL; p++) {
	   double dtmp = 0.;
	  for (m = 0; m < NVEL; m++) {
	    dtmp += mi_cd[p][m]*mode[m];
	  }
	  f_d[nsite*p + index] = dtmp;
	  }
  
  
  return;
}



__device__ void collision_binary_lb_site_gpu_d(const double* __restrict__ force_global_d, 
					  double* __restrict__ f_d, 
					  const double* __restrict__ ftmp_d, 
					  const char* __restrict__ site_map_status_d, 
					  const double* __restrict__ phi_site_d,		
					  const double* __restrict__ grad_phi_site_d,	
					  const double* __restrict__ delsq_phi_site_d,	
					  const double* __restrict__ force_d, 
					       double* __restrict__ velocity_d, const int nsite, const int index) 
{
  int       p, m;                    /* velocity index */
  int       i, j;                    /* summed over indices ("alphabeta") */

  double    mode[NVEL];              /* Modes; hydrodynamic + ghost */
  double    rho, rrho;               /* Density, reciprocal density */
  double    u[3];                    /* Velocity */
  double    s[3][3];                 /* Stress */
  double    seq[3][3];               /* equilibrium stress */
  double    shat[3][3];              /* random stress */
  double    ghat[NVEL];              /* noise for ghosts */

  double    force[3];                /* External force */
  double    tr_s, tr_seq;

  double    force_local[3];

  double f_loc[2*NVEL]; /* thread local copy of f_ data */

  const double   r3     = (1.0/3.0);

  double    phi, jdotc, sphidotq;    /* modes */
  double    jphi[3];
  double    sth[3][3], sphi[3][3];
  double    mu;                      /* Chemical potential */
  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */


  /* ndist is always 2 in this routine. Use of hash define may help compiler */
#define NDIST 2



  /* load data into registers */
  for(p = 0; p < NVEL; p++) {
    for(m = 0; m < NDIST; m++) {
      f_loc[NVEL*m+p] = ftmp_d[nsite*NDIST*p + nsite*m + index];
    }
  }
  
  
  /* Compute all the modes */
  for (m = 0; m < NVEL; m++) {
    double mode_tmp = 0.0;
    for (p = 0; p < NVEL; p++) {
      mode_tmp += f_loc[p]*ma_cd[m][p];
    }
    mode[m] = mode_tmp;
  }
  
  
  /* For convenience, write out the physical modes. */
  
  rho = mode[0];
  for (i = 0; i < 3; i++) {
    u[i] = mode[1 + i];
  }
  s[X][X] = mode[4];
  s[X][Y] = mode[5];
  s[X][Z] = mode[6];
  s[Y][X] = s[X][Y];
  s[Y][Y] = mode[7];
  s[Y][Z] = mode[8];
  s[Z][X] = s[X][Z];
  s[Z][Y] = s[Y][Z];
  s[Z][Z] = mode[9];
  
  /* Compute the local velocity, taking account of any body force */
  
  rrho = 1.0/rho;
  /* hydrodynamics_get_force_local(index, force_local); */
  for (i = 0; i < 3; i++) {
    force_local[i] = force_d[i*nsite+index];
  }
  
  for (i = 0; i < 3; i++) {
    force[i] = (force_global_d[i] + force_local[i]);
    u[i] = rrho*(u[i] + 0.5*force[i]);
  }
  /* hydrodynamics_set_velocity(index, u); */
  for (i = 0; i < 3; i++) {
    velocity_d[i*nsite+index] = u[i];
  }
  
  /* Compute the thermodynamic component of the stress */
  
  symmetric_chemical_stress_gpu_d(index, sth, phi_site_d,
				  grad_phi_site_d,
				  delsq_phi_site_d,nsite);
  
  /* Relax stress with different shear and bulk viscosity */
  
  tr_s   = 0.0;
  tr_seq = 0.0;
  
  for (i = 0; i < 3; i++) {
    /* Set equilibrium stress, which includes thermodynamic part */
    for (j = 0; j < 3; j++) {
      seq[i][j] = rho*u[i]*u[j] + sth[i][j];
    }
    /* Compute trace */
    tr_s   += s[i][i];
    tr_seq += seq[i][i];
  }
  
  /* Form traceless parts */
  for (i = 0; i < 3; i++) {
    s[i][i]   -= r3*tr_s;
    seq[i][i] -= r3*tr_seq;
  }
  
  /* Relax each mode */
  tr_s = tr_s - rtau_bulk_d*(tr_s - tr_seq);
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      s[i][j] -= rtau_shear_d*(s[i][j] - seq[i][j]);
      s[i][j] += d_cd[i][j]*r3*tr_s;
      
      /* Correction from body force (assumes equal relaxation times) */
      
      s[i][j] += (2.0-rtau_shear_d)*(u[i]*force[j] + force[i]*u[j]);
      shat[i][j] = 0.0;
    }
  }
  
  //if (isothermal_fluctuations_) fluctuations_on(shat, ghat);
  
  /* Now reset the hydrodynamic modes to post-collision values */
  
  mode[1] = mode[1] + force[X];    /* Conserved if no force */
  mode[2] = mode[2] + force[Y];    /* Conserved if no force */
  mode[3] = mode[3] + force[Z];    /* Conserved if no force */
  mode[4] = s[X][X] + shat[X][X];
  mode[5] = s[X][Y] + shat[X][Y];
  mode[6] = s[X][Z] + shat[X][Z];
  mode[7] = s[Y][Y] + shat[Y][Y];
  mode[8] = s[Y][Z] + shat[Y][Z];
  mode[9] = s[Z][Z] + shat[Z][Z];
  
  /* Ghost modes are relaxed toward zero equilibrium. */
  
  for (m = NHYDRO; m < NVEL; m++) {
    mode[m] = mode[m] - rtau_d[m]*(mode[m] - 0.0) + ghat[m];
  }
  
  /* Project post-collision modes back onto the distribution */
  
  
  double f_tmp;
  
  for (p = 0; p < NVEL; p++) {
    f_tmp = 0.0;
    for (m = 0; m < NVEL; m++) {
      f_tmp += mi_cd[p][m]*mode[m];
    }
    f_d[nsite*NDIST*p + index] = f_tmp;
  }
  
  /* Now, the order parameter distribution */
  
  phi = phi_site_d[index];
  mu = symmetric_chemical_potential_gpu_d(index, phi_site_d,
					  delsq_phi_site_d);
  
  jphi[X] = 0.0;
  jphi[Y] = 0.0;
  jphi[Z] = 0.0;
  for (p = 1; p < NVEL; p++) {
    for (i = 0; i < 3; i++) {
      jphi[i] += f_loc[NVEL + p]*cv_cd[p][i];
    }
  }
  
  /* Relax order parameters modes. See the comments above. */
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      sphi[i][j] = phi*u[i]*u[j] + mu*d_cd[i][j];
    }
    jphi[i] = jphi[i] - rtau2_d*(jphi[i] - phi*u[i]);
  }
  
  /* Now update the distribution */
	
  for (p = 0; p < NVEL; p++) {
    
    int dp0 = (p == 0);
    jdotc    = 0.0;
    sphidotq = 0.0;
    
    for (i = 0; i < 3; i++) {
      jdotc += jphi[i]*cv_cd[p][i];
      for (j = 0; j < 3; j++) {
	sphidotq += sphi[i][j]*q_cd[p][i][j];
      }
    }
    
    /* Project all this back to the distributions. The magic
     * here is to move phi into the non-propagating distribution. */
    
    f_d[nsite*NDIST*p+nsite+index]
      = wv_cd[p]*(jdotc*rcs2_d + sphidotq*r2rcs4) + phi*dp0;
    
  }
  
  


  return;
}


/* pack edges on the accelerator */
__global__ static void collision_edge_gpu_d(int nhalo, 
						   int N[3],
						   const double* __restrict__ force_global_d, 
						   double* __restrict__ f_d, 
					    const double* __restrict__ ftmp_d, 
						   const char* __restrict__ site_map_status_d, 
						   const double* __restrict__ phi_site_d,		
						   const double* __restrict__ grad_phi_site_d,	
						   const double* __restrict__ delsq_phi_site_d,	
						   const double* __restrict__ force_d, 
					    double* __restrict__ velocity_d,int colltype, int dirn)
{



  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];
 
  int Nedge[3];

  if (dirn == X){
    Nedge[X]=nhalo;
    Nedge[Y]=N[Y];
    Nedge[Z]=N[Z];
  }
  else if (dirn == Y){
    //Nedge[X]=Nall[X];
    Nedge[X]=N[X]-2*nhalo;
    Nedge[Y]=nhalo;
    Nedge[Z]=N[Z];
  }
  else if (dirn == Z){
    Nedge[X]=N[X]-2*nhalo;
    Nedge[Y]=N[Y]-2*nhalo;
    Nedge[Z]=nhalo;
  }


  int p,m, index,ii,jj,kk,ii_,jj_,kk_;
 
  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  

  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);
      
      /* LOW EDGE */
      if (dirn == X){
	index = get_linear_index_gpu_d(ii+nhalo,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn == Y){
	index = get_linear_index_gpu_d(ii+2*nhalo,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn == Z){
	index = get_linear_index_gpu_d(ii+2*nhalo,jj+2*nhalo,kk+nhalo,Nall);
      }


      get_coords_from_index_gpu_d(&ii_,&jj_,&kk_,index,Nall);
      /* printf("low dir=%d %d %d %d\n",dirn,ii_,jj_,kk_); */

      if (colltype==BINARY){
      collision_binary_lb_site_gpu_d(force_global_d,
      					      f_d, ftmp_d,
      					      site_map_status_d,
      					       phi_site_d,
      					       grad_phi_site_d,
      					       delsq_phi_site_d,
      							 force_d,
      				     velocity_d, nsite, index);
	}
      else if (colltype==MULTIRELAXATION)
	{

	  
	  collision_multirelaxation_site_gpu_d(force_global_d, f_d, ftmp_d, 
					       site_map_status_d, 
					       force_d, velocity_d, nsite, 
					       index);

	}

 
      /* printf("%f\n",f_d[index]); */

      //if (jj_==1 && kk_==1)printf("X low %d %d %d %f\n",ii_,jj_,kk_,f_d[index]);


        
      /* HIGH EDGE */
      if (dirn == X){
	index = get_linear_index_gpu_d(Nall[X]-2*nhalo+ii,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn == Y){
	index = get_linear_index_gpu_d(ii+2*nhalo,Nall[Y]-2*nhalo+jj,kk+nhalo,Nall);
      }
      else if (dirn == Z){
	index = get_linear_index_gpu_d(ii+2*nhalo,jj+2*nhalo,Nall[Z]-2*nhalo+kk,Nall);
      }

      //printf("high dir=%d %d %d %d\n",dirn,ii,jj,kk);

      if (colltype==BINARY){
      collision_binary_lb_site_gpu_d(force_global_d,
      					      f_d, ftmp_d,
      					      site_map_status_d,
      					       phi_site_d,
      					       grad_phi_site_d,
      					       delsq_phi_site_d,
      							 force_d,
      				     velocity_d, nsite, index);
	}
      else if (colltype==MULTIRELAXATION)
	{

	  
	  collision_multirelaxation_site_gpu_d(force_global_d, f_d, ftmp_d, 
					       site_map_status_d, 
					       force_d, velocity_d, nsite, 
					       index);

	}


    }
  
  
}



__global__ void collision_lb_gpu_d(int ndist, int nhalo, int N[3], 
					  const double* __restrict__ force_global_d, 
					  double* __restrict__ f_d, 
					  const double* __restrict__ ftmp_d, 
					  const char* __restrict__ site_map_status_d, 
					  const double* __restrict__ phi_site_d,		
					  const double* __restrict__ grad_phi_site_d,	
					  const double* __restrict__ delsq_phi_site_d,	
					  const double* __restrict__ force_d, 
				   double* __restrict__ velocity_d, int colltype, int latchunk) 
{

  int       index;                   /* site indices */
  int       p, m;                    /* velocity index */
  int       i, j;                    /* summed over indices ("alphabeta") */

  double    mode[NVEL];              /* Modes; hydrodynamic + ghost */
  double    rho, rrho;               /* Density, reciprocal density */
  double    u[3];                    /* Velocity */
  double    s[3][3];                 /* Stress */
  double    seq[3][3];               /* equilibrium stress */
  double    shat[3][3];              /* random stress */
  double    ghat[NVEL];              /* noise for ghosts */

  double    force[3];                /* External force */
  double    tr_s, tr_seq;

  double    force_local[3];

  double f_loc[2*NVEL]; /* thread local copy of f_ data */

  const double   r3     = (1.0/3.0);

  double    phi, jdotc, sphidotq;    /* modes */
  double    jphi[3];
  double    sth[3][3], sphi[3][3];
  double    mu;                      /* Chemical potential */
  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */


  int threadIndex, nsite, Nall[3], ii, jj, kk, xfac, yfac, ii_,jj_,kk_;

  /* ndist is always 2 in this routine. Use of hash define may help compiler */
#define NDIST 2


  fluctuations_off_gpu_d(shat, ghat); 

  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsite = Nall[X]*Nall[Y]*Nall[Z];

  int N_[3];



  if (latchunk==BULK){
    N_[X]=N[X]-2*nhalo;  N_[Y]=N[Y]-2*nhalo;  N_[Z]=N[Z]-2*nhalo;    
  }
  else{
    N_[X]=N[X];  N_[Y]=N[Y];  N_[Z]=N[Z];
  }

  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  /* Avoid going beyond problem domain */
  if (threadIndex < N_[X]*N_[Y]*N_[Z])
    {


	get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,N_);


      if (latchunk==BULK){
	index = get_linear_index_gpu_d(ii+2*nhalo,jj+2*nhalo,kk+2*nhalo,Nall);
      }
      else{
	index = get_linear_index_gpu_d(ii+nhalo,jj+nhalo,kk+nhalo,Nall);
      }


      if (colltype==BINARY){
      collision_binary_lb_site_gpu_d(force_global_d,
      					      f_d, ftmp_d,
      					      site_map_status_d,
      					       phi_site_d,
      					       grad_phi_site_d,
      					       delsq_phi_site_d,
      							 force_d,
      				     velocity_d, nsite, index);
      }
      else if (colltype==MULTIRELAXATION)
	{

	  
	  collision_multirelaxation_site_gpu_d(force_global_d, f_d, ftmp_d, 
					       site_map_status_d, 
					       force_d, velocity_d, nsite, 
					       index);

	}



      //      } 
      
    }
  
  return;
}

/*****************************************************************************
 *
 *  collision_relaxation_times_set
 *
 *  Note there is an extra normalisation in the lattice fluctuations
 *  which would otherwise give effective kT = cs2
 *
 *****************************************************************************/

void collision_relaxation_times_set_gpu(void) {

  int p;
  double kt;
  double tau_s;
  double tau_b;
  double tau_g;

  extern int is_propagation_ode(void);
 
  if (is_propagation_ode()) {
    rtau_shear = 1.0 / (3.0*get_eta_shear());
    rtau_bulk  = 1.0 / (3.0*get_eta_bulk());
  }
  else {
    rtau_shear = 2.0 / (1.0 + 6.0*get_eta_shear());
    rtau_bulk  = 2.0 / (1.0 + 6.0*get_eta_bulk());
  }

  /* Initialise the relaxation times */

  if (nrelax_ == RELAXATION_M10) {
    for (p = NHYDRO; p < NVEL; p++) {
      rtau_[p] = 1.0;
    }
  }

  if (nrelax_ == RELAXATION_BGK) {
    for (p = 0; p < NVEL; p++) {
      rtau_[p] = rtau_shear;
    }
  }

  if (nrelax_ == RELAXATION_TRT) {

    assert(NVEL != 9);

    tau_g = 2.0/(1.0 + (3.0/8.0)*rtau_shear);

    if (NVEL == 15) {
      rtau_[10] = rtau_shear;
      rtau_[11] = tau_g;
      rtau_[12] = tau_g;
      rtau_[13] = tau_g;
      rtau_[14] = rtau_shear;
    }

    if (NVEL == 19) {
      rtau_[10] = rtau_shear;
      rtau_[14] = rtau_shear;
      rtau_[18] = rtau_shear;

      rtau_[11] = tau_g;
      rtau_[12] = tau_g;
      rtau_[13] = tau_g;
      rtau_[15] = tau_g;
      rtau_[16] = tau_g;
      rtau_[17] = tau_g;
    }
  }

  if (isothermal_fluctuations_) {

    tau_s = 1.0/rtau_shear;
    tau_b = 1.0/rtau_bulk;

    /* Initialise the stress variances */

    kt = fluid_kt();
    kt = kt*rcs2; /* Without normalisation kT = cs^2 */

    var_bulk =
      sqrt(kt)*sqrt(2.0/9.0)*sqrt((tau_b + tau_b - 1.0)/(tau_b*tau_b));
    var_shear =
      sqrt(kt)*sqrt(1.0/9.0)*sqrt((tau_s + tau_s - 1.0)/(tau_s*tau_s));

    /* Noise variances */

    for (p = NHYDRO; p < NVEL; p++) {
      tau_g = 1.0/rtau_[p];
      noise_var[p] =
	sqrt(kt/norm_[p])*sqrt((tau_g + tau_g - 1.0)/(tau_g*tau_g));
    }
  }

  return;
}



/*****************************************************************************
 *
 *  copy_constants_to_gpu
 *
 *  copy constants to accelerator (constant on-chip read-only memory)
 *
 *****************************************************************************/

  void   copy_constants_to_gpu(){

    double a_,b_,kappa_;

    RUN_get_double_parameter("A", &a_);
    RUN_get_double_parameter("B", &b_);
    RUN_get_double_parameter("K", &kappa_);

   /* copy constant values to accelerator (on-chip read-only memory) */
    cudaMemcpyToSymbol(rtau_shear_d, &rtau_shear, sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(rtau_bulk_d, &rtau_bulk, sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(rtau_d, rtau_, NVEL*sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(wv_cd, wv, NVEL*sizeof(double), 0,	
		       cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(ma_cd, ma_, NVEL*NVEL*sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mi_cd, mi_, NVEL*NVEL*sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
     cudaMemcpyToSymbol(d_cd, d_, 3*3*sizeof(double), 0,	
    		       cudaMemcpyHostToDevice);
     cudaMemcpyToSymbol(cv_cd, cv, NVEL*3*sizeof(int), 0,
			cudaMemcpyHostToDevice);
     cudaMemcpyToSymbol(q_cd, q_, NVEL*3*3*sizeof(double), 0,
			cudaMemcpyHostToDevice);
     cudaMemcpyToSymbol(rtau2_d, &rtau2, sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(rcs2_d, &rcs2, sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(a_d, &a_, sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_d, &b_, sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kappa_d, &kappa_, sizeof(double), 0,	
		       cudaMemcpyHostToDevice);
  }


/*****************************************************************************
 *
 *  fluctuations_off_gpu_d
 *
 *  Return zero fluctuations for stress (shat) and ghost (ghat) modes.
 *
 *****************************************************************************/

__device__ void fluctuations_off_gpu_d(double shat[3][3], double ghat[NVEL]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      shat[ia][ib] = 0.0;
    }
  }

  for (ia = NHYDRO; ia < NVEL; ia++) {
    ghat[ia] = 0.0;
  }

  return;
}



/****************************************************************************
 *
 *  symmetric_chemical_stress
 *
 *  Return the chemical stress tensor for given position index.
 *
 *  P_ab = [1/2 A phi^2 + 3/4 B phi^4 - kappa phi \nabla^2 phi
 *       -  1/2 kappa (\nbla phi)^2] \delta_ab
 *       +  kappa \nalba_a phi \nabla_b phi
 *
 ****************************************************************************/
__device__ void symmetric_chemical_stress_gpu_d(const int index, 
						double s[3][3],
						const double* __restrict__ phi_site_d, 
						const double* __restrict__ grad_phi_site_d, 
						const double* __restrict__ delsq_phi_site_d,
						int nsite) {

  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  phi = phi_site_d[index];
  delsq_phi = delsq_phi_site_d[index];
  for (ia = 0; ia < 3; ia++) grad_phi[ia]=grad_phi_site_d[ia*nsite+index];

  p0 = 0.5*a_d*phi*phi + 0.75*b_d*phi*phi*phi*phi
    - kappa_d*phi*delsq_phi - 
    0.5*kappa_d*dot_product_gpu_d(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_cd[ia][ib]	+ kappa_d*grad_phi[ia]*grad_phi[ib];
    }
  }

  return;
}


/****************************************************************************
 *
 *  symmetric_chemical_potential
 *
 *  The chemical potential \mu = \delta F / \delta \phi
 *                             = a\phi + b\phi^3 - \kappa\nabla^2 \phi
 *
 ****************************************************************************/

__device__ double symmetric_chemical_potential_gpu_d(const int index, 
		const double* __restrict__  phi_site_d, const double* __restrict__  delsq_phi_site_d) {

  double mu, phi;

  phi = phi_site_d[index];
  mu = a_d*phi + b_d*phi*phi*phi - kappa_d*delsq_phi_site_d[index];

  return mu;
}


/*****************************************************************************
 *
 *  dot_product
 *
 *****************************************************************************/

__device__ double dot_product_gpu_d(const double a[3], const double b[3]) {

	return (a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z]);
}


/* get 3d coordinates from the index on the accelerator */
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;
  
  *ii = index/xfac;
  *jj = ((index-xfac*(*ii))/yfac);
  *kk = (index-(*ii)*xfac-(*jj)*yfac);

  return;

}


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3])
{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}
