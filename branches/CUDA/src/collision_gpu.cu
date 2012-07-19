/*****************************************************************************
 *
 *  collision_gpu.c
 *
 *  Collision stage routines and associated data.
 *
 *  Isothermal fluctuations following Adhikari et al., Europhys. Lett
 *  (2005).
 *
 *  The relaxation times can be set to give either 'm10', BGK or
 *  'two-relaxation' time (TRT) models.
 *
 *  $Id: collision.c 1728 2012-07-18 08:41:51Z agray3 $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *  Adapted to run on GPU: Alan Gray
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>


/* below define needed to stop repeated declaration of distribution_ndist */
#define INCLUDING_FROM_GPU_SOURCE 

#include "model.h"
#include "site_map.h"
#include "collision_gpu.h"
#include "collision.h"
#include "fluctuations.h"

//#include "free_energy.h"
//#include "phi_cahn_hilliard.h"

#include "control.h"
#include "propagation_ode.h"

static int nmodes_ = NVEL;               /* Modes to use in collsion stage */
static int nrelax_ = RELAXATION_M10;     /* [RELAXATION_M10|TRT|BGK] */
                                         /* Default is M10 */
static int isothermal_fluctuations_ = 0; /* Flag for noise. */

static double rtau_shear;       /* Inverse relaxation time for shear modes */
static double rtau_bulk;        /* Inverse relaxation time for bulk modes */
static double var_shear;        /* Variance for shear mode fluctuations */
static double var_bulk;         /* Variance for bulk mode fluctuations */
static double rtau_[NVEL];      /* Inverse relaxation times */
static double noise_var[NVEL];  /* Noise variances */

static fluctuations_t * fl_;

static double rtau2;

/* static void collision_multirelaxation(void); */
/* static void collision_binary_lb(void); */
/* static void collision_bgk(void); */

/* static void fluctuations_off(double shat[3][3], double ghat[NVEL]); */
/*        void collision_fluctuations(int index, double shat[3][3], */
/* 				   double ghat[NVEL]); */


/*****************************************************************************
 *
 *  collide
 *
 *  Driver routine for the collision stage.
 *
 *  Note that the ODE propagation currently uses ndist == 2, as
 *  well as the LB binary, hence the logic.
 *
 *****************************************************************************/

/* void collide(void) { */

/*   int ndist; */

/*   ndist = distribution_ndist(); */
/*   collision_relaxation_times_set(); */

/*   if ((ndist == 1 || is_propagation_ode() == 1 ) && nrelax_ == RELAXATION_M10) collision_multirelaxation(); */
/*   if  (ndist == 2 && is_propagation_ode() == 0) collision_binary_lb(); */
/*   if ((ndist == 1 || is_propagation_ode() == 1 ) && nrelax_ == RELAXATION_BGK) collision_bgk(); */

/*   return; */
/* } */

void collide_gpu(void) {

  int ndist,nhalo;
  double mobility;
  int N[3];

  ndist = distribution_ndist();
  nhalo = coords_nhalo();
  coords_nlocal(N); 


  //collision_relaxation_times_set();
  //we need a duplicate GPU copy of this routine, as things are set up, 
  // to set the right copy of static (file scope) variables
  collision_relaxation_times_set_gpu();

  mobility = phi_cahn_hilliard_mobility();
  rtau2 = 2.0 / (1.0 + 2.0*mobility);


/*   if ((ndist == 1 || is_propagation_ode() == 1 ) && nrelax_ == RELAXATION_M10) collision_multirelaxation(); */
/*   if  (ndist == 2 && is_propagation_ode() == 0) collision_binary_lb(); */
/*   if ((ndist == 1 || is_propagation_ode() == 1 ) && nrelax_ == RELAXATION_BGK) collision_bgk(); */

/* copy constants to accelerator (constant on-chip read-only memory) */
  copy_constants_to_gpu();
  
  /* set up CUDA grid */
  /* 1D decomposition - use x grid and block dimension only */ 
  int nblocks=(N[X]*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;

  if ((ndist == 1 || is_propagation_ode() == 1 ) && nrelax_ == RELAXATION_M10)

    {
      
      printf("a\n");
    }

  if  (ndist == 2 && is_propagation_ode() == 0) 
    { 

      collision_binary_lb_gpu_d<<<nblocks,DEFAULT_TPB>>>(ndist, nhalo, N_d, 					      force_global_d, 
					      f_d, 
					      site_map_status_d, 
					       phi_site_d,		
					       grad_phi_site_d,	
					       delsq_phi_site_d,	
					      force_d, 
    			       		      velocity_d, 
					      ma_d, 
					      d_d, 
					      mi_d, 
					      cv_d, 
					       q_d, 
					       wv_d);
   

    }

  if ((ndist == 1 || is_propagation_ode() == 1 ) && nrelax_ == RELAXATION_BGK)
    {
      printf("c\n");


    }

    cudaThreadSynchronize();

  return;
}



__global__ void collision_binary_lb_gpu_d(int ndist, int nhalo, int N[3], 
					  double* force_global_d, 
					  double* f_d, 
					  char* site_map_status_d, 
					  double* phi_site_d,		
					  double* grad_phi_site_d,	
					  double* delsq_phi_site_d,	
					  double* force_ptr, 
					  double* velocity_ptr, 
					  double* ma_ptr, 
					  double* d_ptr, 
					  double* mi_ptr, 
					  int* cv_ptr, 
					  double* q_ptr, 
					  double* wv_d) 
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

 /* cast dummy gpu memory pointers to pointers of right type (for 
   * multidimensional arrays) */
  double (*force_d)[3] = (double (*)[3]) force_ptr;
  double (*velocity_d)[3] = (double (*)[3]) velocity_ptr;
  double (*ma_d)[NVEL] = (double (*)[NVEL]) ma_ptr;
  double (*mi_d)[NVEL] = (double (*)[NVEL]) mi_ptr;
  double (*d_d)[3] = (double (*)[3]) d_ptr;
  int (*cv_d)[3] = (int (*)[3]) cv_ptr;
  double (*q_d)[3][3] = (double (*)[3][3]) q_ptr;

  int threadIndex, nsite, Nall[3], ii, jj, kk, xfac, yfac;

  /* ndist is always 2 in this routine. Use of hash define may help compiler */
#define NDIST 2

  //chemical_potential = fe_chemical_potential_function();
  // chemical_stress = fe_chemical_stress_function();


   fluctuations_off_gpu_d(shat, ghat); 

  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsite = Nall[X]*Nall[Y]*Nall[Z];

  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  /* Avoid going beyond problem domain */
  if (threadIndex < N[X]*N[Y]*N[Z])
    {
      
      /* calculate index from CUDA thread index */
      yfac = N[Z];
      xfac = N[Y]*yfac;
      
      ii = threadIndex/xfac;
      jj = ((threadIndex-xfac*ii)/yfac);
      kk = (threadIndex-ii*xfac-jj*yfac);
      
      index = get_linear_index_gpu_d(ii+1,jj+1,kk+1,Nall);
      
/*       if (site_map_status_d[index] == FLUID) */
/* 	{ */
	  
	  
	  /* load data into registers */
	  for(p = 0; p < NVEL; p++) {
	    for(m = 0; m < NDIST; m++) {
	      f_loc[NVEL*m+p] = f_d[nsite*NDIST*p + nsite*m + index];
	    }
	  }
	  
	  
	  /* Compute all the modes */
	  for (m = 0; m < NVEL; m++) {
	    double mode_tmp = 0.0;
	    for (p = 0; p < NVEL; p++) {
	      mode_tmp += f_loc[p]*ma_d[m][p];
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
	    force_local[i] = force_d[index][i];
	  }
	  
	  for (i = 0; i < 3; i++) {
	    force[i] = (force_global_d[i] + force_local[i]);
	    u[i] = rrho*(u[i] + 0.5*force[i]);
	  }
	  /* hydrodynamics_set_velocity(index, u); */
	  for (i = 0; i < 3; i++) {
	    velocity_d[index][i] = u[i];
	  }
	  
	  /* Compute the thermodynamic component of the stress */
	  
	  symmetric_chemical_stress_gpu_d(index, sth, phi_site_d,
					  grad_phi_site_d,
					  delsq_phi_site_d,d_d);

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
	      s[i][j] += d_d[i][j]*r3*tr_s;
	      
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

	  /* the below syncthreads is required, otherwise the above
	     summation goes wrong. This is NOT UNDERSTOOD yet and under
	     investigation - Alan Gray */
	  __syncthreads();

  	  double f_tmp;
	  
 	  for (p = 0; p < NVEL; p++) {
 	    f_tmp = 0.0;
 	    for (m = 0; m < NVEL; m++) {
 	      f_tmp += mi_d[p][m]*mode[m];
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
	    jphi[i] += f_loc[NVEL + p]*cv_d[p][i];
	  }
	}
	
	/* Relax order parameters modes. See the comments above. */
	
	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    sphi[i][j] = phi*u[i]*u[j] + mu*d_d[i][j];
	  }
	  jphi[i] = jphi[i] - rtau2_d*(jphi[i] - phi*u[i]);
	}
	
	/* Now update the distribution */
	
	for (p = 0; p < NVEL; p++) {
	  
	  int dp0 = (p == 0);
	  jdotc    = 0.0;
	  sphidotq = 0.0;
	  
	  for (i = 0; i < 3; i++) {
	    jdotc += jphi[i]*cv_d[p][i];
	    for (j = 0; j < 3; j++) {
	      sphidotq += sphi[i][j]*q_d[p][i][j];
	    }
	  }
	  
	  /* Project all this back to the distributions. The magic
	   * here is to move phi into the non-propagating distribution. */
	  
	  f_d[nsite*NDIST*p+nsite+index]
	    = wv_d[p]*(jdotc*rcs2_d + sphidotq*r2rcs4) + phi*dp0;
	  
	}
	
	
/* 	} */
      
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
						double *phi_site_d, 
						double *grad_phi_site_d, 
						double *delsq_phi_site_d,
						double d_d[3][3]) {

  int ia, ib;
  double phi;
  double delsq_phi;
  double grad_phi[3];
  double p0;

  phi = phi_site_d[index];
  delsq_phi = delsq_phi_site_d[index];
  for (ia = 0; ia < 3; ia++) grad_phi[ia]=grad_phi_site_d[3*index+ia];

  p0 = 0.5*a_d*phi*phi + 0.75*b_d*phi*phi*phi*phi
    - kappa_d*phi*delsq_phi - 
    0.5*kappa_d*dot_product_gpu_d(grad_phi, grad_phi);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = p0*d_d[ia][ib]	+ kappa_d*grad_phi[ia]*grad_phi[ib];
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
		double *phi_site_d, double *delsq_phi_site_d) {

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


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3])
{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}
