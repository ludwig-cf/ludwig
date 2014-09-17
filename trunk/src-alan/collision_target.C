

#define INCLUDED_FROM_TARGET

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "targetDP.h"
#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "lattice.h"
#include "model.h"
#include "site_map.h"
#include "collision.h"
#include "fluctuations.h"

#include "phi.h"
#include "free_energy.h"
#include "phi_cahn_hilliard.h"

#include "control.h"
#include "propagation_ode.h"

#define HOST
#ifdef CUDA
#define HOST extern "C"
#endif


static int nmodes_ = NVEL;               /* Modes to use in collsion stage */
static int nrelax_ = RELAXATION_M10;     /* [RELAXATION_M10|TRT|BGK] */
                                         /* Default is M10 */


// static int isothermal_fluctuations_ = 0; /* Flag for noise. */

// static double rtau_shear;       /* Inverse relaxation time for shear modes */
// static double rtau_bulk;        /* Inverse relaxation time for bulk modes */
// static double var_shear;        /* Variance for shear mode fluctuations */
// static double var_bulk;         /* Variance for bulk mode fluctuations */
// static double rtau_[NVEL];      /* Inverse relaxation times */
// static double noise_var[NVEL];  /* Noise variances */

// static fluctuations_t * fl_;

extern int isothermal_fluctuations_ = 0; /* Flag for noise. */

extern double rtau_shear;       /* Inverse relaxation time for shear modes */
extern double rtau_bulk;        /* Inverse relaxation time for bulk modes */
extern double var_shear;        /* Variance for shear mode fluctuations */
extern double var_bulk;         /* Variance for bulk mode fluctuations */
extern double rtau_[NVEL];      /* Inverse relaxation times */
extern double noise_var[NVEL];  /* Noise variances */

extern fluctuations_t * fl_;


static void collision_multirelaxation(void);
static void collision_binary_lb(void);
static void collision_bgk(void);

HOST void fluctuations_off(double shat[3][3], double ghat[NVEL]);
       void collision_fluctuations(int index, double shat[3][3],
				   double ghat[NVEL]);

/* Constants*/

TARGET_CONST int tc_nSites;
TARGET_CONST double tc_rtau_shear;
TARGET_CONST double tc_rtau_bulk;
TARGET_CONST double tc_rtau[NVEL];
TARGET_CONST double tc_wv[NVEL];
TARGET_CONST double tc_ma[NVEL][NVEL];
TARGET_CONST double tc_mi[NVEL][NVEL];
TARGET_CONST int tc_cv[NVEL][3];
TARGET_CONST double tc_rtau2;
TARGET_CONST double tc_rcs2;
TARGET_CONST double tc_force_global[3];
TARGET_CONST double tc_d[3][3];
TARGET_CONST double tc_q[NVEL][3][3];

// type for chemical potential function
// TODO This is generic fine here
//typedef double (*mu_fntype)(const int, const int, const double*, const double*);


HOST void get_chemical_stress_target(pth_fntype* h_chemical_stress);
HOST void get_chemical_potential_target(mu_fntype* h_chemical_potential);

TARGET void collision_binary_lb_site( double* __restrict__ t_f, 
				      const double* __restrict__ t_force, 
				      double* __restrict__ t_velocity,
				      double* __restrict__ t_phi,
				      double* __restrict__ t_gradphi,
				      double* __restrict__ t_delsqphi,
				      pth_fntype* chemical_stress,
				      mu_fntype* chemical_potential,
				      const int baseIndex){

  int       p, m;                    /* velocity index */
  int       i, j;                    /* summed over indices ("alphabeta") */

  double    mode[NVEL*NILP];              /* Modes; hydrodynamic + ghost */
  double    rho[NILP], rrho[NILP];               /* Density, reciprocal density */
  double    uloc[3*NILP];                    /* Velocity */
  double    s[3][3*NILP];                 /* Stress */
  double    seq[3][3*NILP];               /* equilibrium stress */
  double    shat[3][3*NILP];              /* random stress */
  double    ghat[NVEL*NILP];              /* noise for ghosts */

  double    force[3*NILP];                /* External force */
  double    tr_s[NILP], tr_seq[NILP];

  const double   r3     = (1.0/3.0);

  double    phi[NILP], jdotc[NILP], sphidotq[NILP];    /* modes */
  double    jphi[3*NILP];
  double    sth[3][3*NILP], sphi[3][3*NILP];
  double    mu[NILP];                      /* Chemical potential */
  //  double    mobility;
  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  //  double (* chemical_potential)(const int index, const int nop);
  //void   (* chemical_stress)(const int index, double s[3][3]);
  
#define NDIST 2 //for binary collision
  
  //TEMP
  int vecIndex=0;
  
  double floc[NVEL*NDIST*NILP];
  

  //TO DO HACK

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      TARGET_ILP(vecIndex) shat[i][ILPIDX(j)] = 0.0;
    }
  }
  
  for (i = NHYDRO; i < NVEL; i++) {
    TARGET_ILP(vecIndex) ghat[ILPIDX(i)] = 0.0;
  }

  
  /* load data */

  for(p = 0; p < NVEL; p++) {
    for(m = 0; m < NDIST; m++) {
      TARGET_ILP(vecIndex) floc[ILPIDX(NVEL*m+p)] = 
	t_f[tc_nSites*NDIST*p + tc_nSites*m + baseIndex + vecIndex];
    }
  }

  
  for (m = 0; m < NVEL; m++) {
    TARGET_ILP(vecIndex) mode[ILPIDX(m)] = 0.0;
    for (p = 0; p < NVEL; p++) {
      TARGET_ILP(vecIndex) mode[ILPIDX(m)] += floc[ILPIDX(p)]*tc_ma[m][p];
    }
    
  }
  

  /* For convenience, write out the physical modes. */
  
  TARGET_ILP(vecIndex) rho[vecIndex] = mode[ILPIDX(0)];
  for (i = 0; i < 3; i++) {
    TARGET_ILP(vecIndex) uloc[ILPIDX(i)] = mode[ILPIDX(1 + i)];
  }

  TARGET_ILP(vecIndex){
    s[X][ILPIDX(X)] = mode[ILPIDX(4)];
    s[X][ILPIDX(Y)] = mode[ILPIDX(5)];
    s[X][ILPIDX(Z)] = mode[ILPIDX(6)];
    s[Y][ILPIDX(X)] = s[X][ILPIDX(Y)];
    s[Y][ILPIDX(Y)] = mode[ILPIDX(7)];
    s[Y][ILPIDX(Z)] = mode[ILPIDX(8)];
    s[Z][ILPIDX(X)] = s[X][ILPIDX(Z)];
    s[Z][ILPIDX(Y)] = s[Y][ILPIDX(Z)];
    s[Z][ILPIDX(Z)] = mode[ILPIDX(9)];  
  }
  /* Compute the local velocity, taking account of any body force */
  
  TARGET_ILP(vecIndex) rrho[vecIndex] = 1.0/rho[vecIndex] ;
  


  
  for (i = 0; i < 3; i++) {	

    //TODO NEED TO TRANSPOSE FORCE, VELOCITY

    TARGET_ILP(vecIndex){
      force[ILPIDX(i)] = (tc_force_global[i] + t_force[(baseIndex+vecIndex)*3+i]);
      uloc[ILPIDX(i)] = rrho[vecIndex]*(uloc[ILPIDX(i)] + 0.5*force[ILPIDX(i)]);  
    }
  }
  
  //      hydrodynamics_set_velocity(baseIndex, u);
  for (i = 0; i < 3; i++) 
    TARGET_ILP(vecIndex) t_velocity[(baseIndex+vecIndex)*3+i]=uloc[ILPIDX(i)];
  
  
  /* Compute the thermodynamic component of the stress */
  
  (*chemical_stress)(baseIndex, sth,  t_phi, t_gradphi, t_delsqphi);

  
  /* Relax stress with different shear and bulk viscosity */
  
  TARGET_ILP(vecIndex) tr_s[vecIndex]  = 0.0;
  TARGET_ILP(vecIndex) tr_seq[vecIndex]  = 0.0;
  
  for (i = 0; i < 3; i++) {
    /* Set equilibrium stress, which includes thermodynamic part */
    for (j = 0; j < 3; j++) {
      TARGET_ILP(vecIndex)
	seq[i][ILPIDX(j)] = rho[vecIndex]*uloc[ILPIDX(i)]*uloc[ILPIDX(j)] + sth[i][ILPIDX(j)];
    }
    /* Compute trace */
    TARGET_ILP(vecIndex){
      tr_s[vecIndex]   += s[i][ILPIDX(i)];
      tr_seq[vecIndex] += seq[i][ILPIDX(i)];
    }
  }
  
  /* Form traceless parts */
  for (i = 0; i < 3; i++) {
    TARGET_ILP(vecIndex){
      s[i][ILPIDX(i)]   -= r3*tr_s[vecIndex];
      seq[i][ILPIDX(i)] -= r3*tr_seq[vecIndex];
    }
  }
  
  /* Relax each mode */
  TARGET_ILP(vecIndex) tr_s[vecIndex] = tr_s[vecIndex] - tc_rtau_bulk*(tr_s[vecIndex]
						  - tr_seq[vecIndex]);
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      TARGET_ILP(vecIndex){
	s[i][ILPIDX(j)] -= tc_rtau_shear*(s[i][ILPIDX(j)] - seq[i][ILPIDX(j)]);
	s[i][ILPIDX(j)] += tc_d[i][j]*r3*tr_s[vecIndex];
	
	/* Correction from body force (assumes equal relaxation times) */
	
	s[i][ILPIDX(j)] += (2.0-tc_rtau_shear)*(uloc[ILPIDX(i)]*force[ILPIDX(j)] + force[ILPIDX(i)]*uloc[ILPIDX(j)]);
	shat[i][ILPIDX(j)] = 0.0;
      }
    }
  }
  


  //if (isothermal_fluctuations_) {
  //	collision_fluctuations(baseIndex, shat, ghat);
  //}
  
  /* Now reset the hydrodynamic modes to post-collision values */
  
  TARGET_ILP(vecIndex){
    mode[ILPIDX(1)] = mode[ILPIDX(1)] + force[ILPIDX(X)];    /* Conserved if no force */
    mode[ILPIDX(2)] = mode[ILPIDX(2)] + force[ILPIDX(Y)];    /* Conserved if no force */
    mode[ILPIDX(3)] = mode[ILPIDX(3)] + force[ILPIDX(Z)];    /* Conserved if no force */
    mode[ILPIDX(4)] = s[X][ILPIDX(X)] + shat[X][ILPIDX(X)];
    mode[ILPIDX(5)] = s[X][ILPIDX(Y)] + shat[X][ILPIDX(Y)];
    mode[ILPIDX(6)] = s[X][ILPIDX(Z)] + shat[X][ILPIDX(Z)];
    mode[ILPIDX(7)] = s[Y][ILPIDX(Y)] + shat[Y][ILPIDX(Y)];
    mode[ILPIDX(8)] = s[Y][ILPIDX(Z)] + shat[Y][ILPIDX(Z)];
    mode[ILPIDX(9)] = s[Z][ILPIDX(Z)] + shat[Z][ILPIDX(Z)];
  }
  
  
  /* Ghost modes are relaxed toward zero equilibrium. */
  
  for (m = NHYDRO; m < NVEL; m++) {
    TARGET_ILP(vecIndex)
    mode[ILPIDX(m)] = mode[ILPIDX(m)] 
     - tc_rtau[m]*(mode[ILPIDX(m)] - 0.0) + ghat[ILPIDX(m)];
  }
  
  
  /* Project post-collision modes back onto the distribution */
  /* matrix multiplication for full SIMD vector */

  double ftmp[NILP];
  
  for (p = 0; p < NVEL; p++) {
    TARGET_ILP(vecIndex) ftmp[vecIndex] = 0.0;
    for (m = 0; m < NVEL; m++) {
      TARGET_ILP(vecIndex) ftmp[vecIndex] += tc_mi[p][m]*mode[ILPIDX(m)];
    }
    TARGET_ILP(vecIndex) t_f[tc_nSites*NDIST*p + baseIndex + vecIndex] = ftmp[vecIndex];
  }
  
  
  /* Now, the order parameter distribution */
  //HACK
  TARGET_ILP(vecIndex){
    phi[vecIndex]=0;
    mu[vecIndex]=0;
  }



  //phi =  t_phi[baseIndex];;
  TARGET_ILP(vecIndex) mu[vecIndex] 
        = (*chemical_potential)(baseIndex+vecIndex, 0, t_phi, t_delsqphi);
  
  TARGET_ILP(vecIndex){
    jphi[ILPIDX(X)] = 0.0;
    jphi[ILPIDX(Y)] = 0.0;
    jphi[ILPIDX(Z)] = 0.0;
  }
  for (p = 1; p < NVEL; p++) {
    for (i = 0; i < 3; i++) {
      TARGET_ILP(vecIndex) jphi[ILPIDX(i)] += floc[ILPIDX(NVEL+p)]*tc_cv[p][i];
    }
  }
  
  
  /* Relax order parameters modes. See the comments above. */
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      TARGET_ILP(vecIndex) sphi[i][ILPIDX(j)] = phi[vecIndex]*uloc[ILPIDX(i)]
  	*uloc[ILPIDX(j)] + mu[vecIndex]*tc_d[i][j];
      /* sphi[i][j] = phi*uloc[i]*uloc[j] + cs2*mobility*mu*d_[i][j];*/
    }
    TARGET_ILP(vecIndex) jphi[ILPIDX(i)] = jphi[ILPIDX(i)] 
      - tc_rtau2*(jphi[ILPIDX(i)] - phi[vecIndex]*uloc[ILPIDX(i)]);
    /* jphi[i] = phi*uloc[i];*/
  }
  
  /* Now update the distribution */
  
  for (p = 0; p < NVEL; p++) {
    
    int dp0 = (p == 0);
    TARGET_ILP(vecIndex){
      jdotc[vecIndex]    = 0.0;
      sphidotq[vecIndex] = 0.0;
    }
    
    for (i = 0; i < 3; i++) {
      TARGET_ILP(vecIndex) jdotc[vecIndex] += jphi[ILPIDX(i)]*tc_cv[p][i];
      for (j = 0; j < 3; j++) {
  	TARGET_ILP(vecIndex) sphidotq[vecIndex] += sphi[i][ILPIDX(j)]*tc_q[p][i][j];
      }
    }
    
    /* Project all this back to the distributions. The magic
     * here is to move phi into the non-propagating distribution. */
    
    TARGET_ILP(vecIndex) t_f[tc_nSites*NDIST*p + tc_nSites + baseIndex + vecIndex] = 
      tc_wv[p]*(jdotc[vecIndex]*tc_rcs2 + sphidotq[vecIndex]*r2rcs4) + phi[vecIndex]*dp0;
  }
  
  return;
  
}



TARGET_ENTRY void collision_binary_lb_lattice( double* __restrict__ t_f, 
					       const double* __restrict__ t_force, 
					       double* __restrict__ t_velocity,
					       double* __restrict__ t_phi,
					       double* __restrict__ t_gradphi,
					       double* __restrict__ t_delsqphi,
					       pth_fntype* chemical_stress,
					       mu_fntype* chemical_potential,
					       const int nSites){

  int tpIndex;
  TARGET_TLP(tpIndex,nSites)
    {
	
      collision_binary_lb_site( t_f, t_force, t_velocity,t_phi,t_gradphi,t_delsqphi,chemical_stress,chemical_potential,tpIndex);

    }

}


// host copies of fields
extern double* f_;
extern double* phi_site;
extern double* phi_delsq_;
extern double* phi_grad_;
extern double* f;
extern double* u;

// target copies of fields 
static double *t_f; 
static double *t_phi; 
static double *t_delsqphi; 
static double *t_gradphi; 
static double *t_force; 
static double *t_velocity; 


void init_fields_target(){

  int N[3];
  coords_nlocal(N);

  int nFields=NVEL*NDIST;
  int nhalo=coords_nhalo();

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;  Nall[Y]=N[Y]+2*nhalo;  Nall[Z]=N[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];



  targetInit(nSites, nFields);

  targetCalloc((void **) &t_f, nSites*nFields*sizeof(double));
  targetCalloc((void **) &t_phi, nSites*sizeof(double));
  targetCalloc((void **) &t_delsqphi, nSites*sizeof(double));
  targetCalloc((void **) &t_gradphi, nSites*3*sizeof(double));
  targetCalloc((void **) &t_force, nSites*3*sizeof(double));
  targetCalloc((void **) &t_velocity, nSites*3*sizeof(double));

  checkTargetError("Binary Collision Allocation");


  
}


void finalise_fields_target(){

  targetFree(t_f);
  targetFree(t_phi);
  targetFree(t_delsqphi);
  targetFree(t_gradphi);
  targetFree(t_force);
  targetFree(t_velocity);

  checkTargetError("Binary Collision Free");

  targetFinalize();


}

void put_fields_on_target_masked(){

  int       ic, jc, kc, index;       /* site indices */

  int N[3];
  coords_nlocal(N);

  int nFields=NVEL*NDIST;
  int nhalo=coords_nhalo();

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;  Nall[Y]=N[Y]+2*nhalo;  Nall[Z]=N[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];




//set up site mask
  char* siteMask = (char*) calloc(nSites,sizeof(char));
  if(!siteMask){
    printf("siteMask malloc failed\n");
    exit(1);
  }

  // set all non-halo sites to 1
  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {
  	index=coords_index(ic, jc, kc);
  	siteMask[index]=1;
      }
    }
  }

  copyToTargetMasked(t_f,f_,nSites,nFields,siteMask); 
  copyToTargetMaskedAoS(t_phi,phi_site,nSites,1,siteMask); 
  copyToTargetMaskedAoS(t_delsqphi,phi_delsq_,nSites,1,siteMask); 
  copyToTargetMaskedAoS(t_gradphi,phi_grad_,nSites,3,siteMask); 
  copyToTargetMaskedAoS(t_force,f,nSites,3,siteMask); 
  copyToTargetMaskedAoS(t_velocity,u,nSites,3,siteMask); 


  free(siteMask);


}


void get_fields_from_target_masked(){

  int       ic, jc, kc, index;       /* site indices */

  int N[3];
  coords_nlocal(N);

  int nFields=NVEL*NDIST;
  int nhalo=coords_nhalo();

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;  Nall[Y]=N[Y]+2*nhalo;  Nall[Z]=N[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];
  

//set up site mask
  char* siteMask = (char*) calloc(nSites,sizeof(char));
  if(!siteMask){
    printf("siteMask malloc failed\n");
    exit(1);
  }

  // set all non-halo sites to 1
  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {
  	index=coords_index(ic, jc, kc);
  	siteMask[index]=1;
      }
    }
  }

  copyFromTargetMasked(f_,t_f,nSites,nFields,siteMask); 
  copyFromTargetMaskedAoS(phi_site,t_phi,nSites,1,siteMask); 
  copyFromTargetMaskedAoS(phi_delsq_,t_delsqphi,nSites,1,siteMask); 
  copyFromTargetMaskedAoS(phi_grad_,t_gradphi,nSites,3,siteMask); 
  copyFromTargetMaskedAoS(f,t_force,nSites,3,siteMask); 
  copyFromTargetMaskedAoS(u,t_velocity,nSites,3,siteMask); 


  free(siteMask);



}


HOST void collision_binary_lb_target() {

  int       N[3];
  int       ic, jc, kc, index;       /* site indices */

  double    shat[3][3];              /* random stress */
  double    ghat[NVEL];              /* noise for ghosts */

  double    force_global[3]; 

  double    rtau2;
  double    mobility;

   double (* chemical_potential)(const int index, const int nop, 
				double* t_phi, double* t_delsq_phi);
  //void   (* chemical_stress)(const int index, double s[3][3]);

#define NDIST 2 //for binary collision


  assert (NDIM == 3);
  coords_nlocal(N);
  fluid_body_force(force_global);


  mu_fntype* t_chemical_potential; 
  targetMalloc((void**) &t_chemical_potential, sizeof(mu_fntype));

  //TODO  
  //chemical_potential = fe_chemical_potential_function();
  //the below is currently hardwired in symmetric module. Need to
  // abstract in fe interface
  get_chemical_potential_target(t_chemical_potential);


  pth_fntype* t_chemical_stress; 
  targetMalloc((void**) &t_chemical_stress, sizeof(pth_fntype));
  get_chemical_stress_target(t_chemical_stress);


  //chemical_stress = fe_chemical_stress_function();

  /* The lattice mobility gives tau = (M rho_0 / Delta t) + 1 / 2,
   * or with rho_0 = 1 etc: (1 / tau) = 2 / (2M + 1) */

  mobility = phi_cahn_hilliard_mobility();
  rtau2 = 2.0 / (1.0 + 2.0*mobility);
  fluctuations_off(shat, ghat);


  // start lattice operation setup


   int nFields=NVEL*NDIST;
   int nhalo=coords_nhalo();

   int Nall[3];
   Nall[X]=N[X]+2*nhalo;  Nall[Y]=N[Y]+2*nhalo;  Nall[Z]=N[Z]+2*nhalo;

   int nSites=Nall[X]*Nall[Y]*Nall[Z];

   init_fields_target();




  put_fields_on_target_masked();
  
  //start constant setup


  copyConstantDoubleToTarget(&tc_rtau_shear, &rtau_shear, sizeof(double)); 
  copyConstantDoubleToTarget(&tc_rtau_bulk, &rtau_bulk, sizeof(double));
  copyConstantDouble1DArrayToTarget(tc_rtau, rtau_, NVEL*sizeof(double)); 
  copyConstantDouble1DArrayToTarget(tc_wv, wv, NVEL*sizeof(double));
  copyConstantDouble2DArrayToTarget( (double **) tc_ma, (double*) ma_, NVEL*NVEL*sizeof(double));
  copyConstantDouble2DArrayToTarget((double **) tc_mi, (double*) mi_, NVEL*NVEL*sizeof(double));
  copyConstantInt2DArrayToTarget((int **) tc_cv,(int*) cv, NVEL*3*sizeof(int)); 
  copyConstantDoubleToTarget(&tc_rtau2, &rtau2, sizeof(double));
  copyConstantDoubleToTarget(&tc_rcs2, &rcs2, sizeof(double));
  copyConstantIntToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstantDouble1DArrayToTarget(tc_force_global,force_global, 3*sizeof(double)); 
  copyConstantDouble2DArrayToTarget((double **) tc_d, (double*) d_, 3*3*sizeof(double));
  copyConstantDouble3DArrayToTarget((double ***) tc_q, (double *)q_, NVEL*3*3*sizeof(double)); 
  checkTargetError("constants");
  //end constant setup


  collision_binary_lb_lattice TARGET_LAUNCH(nSites) ( t_f, t_force, t_velocity,t_phi,t_gradphi,t_delsqphi,t_chemical_stress,t_chemical_potential,nSites);

  syncTarget();
  checkTargetError("Binary Collision Kernel");

  //  end lattice operation


  get_fields_from_target_masked();

  finalise_fields_target();

  targetFree(t_chemical_potential);
  targetFree(t_chemical_stress);




  return;
}


HOST void collision_binary_lb_target2() {



  //---
  int       ic, jc, kc, index;       /* site indices */

  int N[3];
  coords_nlocal(N);

  int nFields=NVEL*NDIST;
  int nhalo=coords_nhalo();

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;  Nall[Y]=N[Y]+2*nhalo;  Nall[Z]=N[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];
  

//set up site mask
  char* siteMask = (char*) calloc(nSites,sizeof(char));
  if(!siteMask){
    printf("siteMask malloc failed\n");
    exit(1);
  }

  // set all non-halo sites to 1
  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {
  	index=coords_index(ic, jc, kc);
  	siteMask[index]=1;
      }
    }
  }

  //---


  //int       N[3];
  //int       ic, jc, kc, index;       /* site indices */





  //coords_nlocal(N);

  //int nFields=NVEL*NDIST;
  //int nhalo=coords_nhalo();


  //int Nall[3];
  //Nall[X]=N[X]+2*nhalo;  Nall[Y]=N[Y]+2*nhalo;  Nall[Z]=N[Z]+2*nhalo;

  //int nSites=Nall[X]*Nall[Y]*Nall[Z];


  double    shat[3][3];              /* random stress */
  double    ghat[NVEL];              /* noise for ghosts */

  double    force_global[3]; 

  double    rtau2;
  double    mobility;

#define NDIST 2 //for binary collision


  assert (NDIM == 3);
  coords_nlocal(N);
  fluid_body_force(force_global);


  mu_fntype* t_chemical_potential; 
  targetMalloc((void**) &t_chemical_potential, sizeof(mu_fntype));

  //TODO  
  //chemical_potential = fe_chemical_potential_function();
  //the below is currently hardwired in symmetric module. Need to
  // abstract in fe interface
  get_chemical_potential_target(t_chemical_potential);

  //chemical_stress = fe_chemical_stress_function();
  pth_fntype* t_chemical_stress; 
  targetMalloc((void**) &t_chemical_stress, sizeof(pth_fntype));
  get_chemical_stress_target(t_chemical_stress);


  //  fluctuations_off(shat, ghat);


  // start lattice operation setup

// host copies of fields
extern double* f_;
extern double* phi_site;
extern double* phi_delsq_;
extern double* phi_grad_;
extern double* f;
extern double* u;

// target copies of fields 
double *t_f; 
double *t_phi; 
double *t_delsqphi; 
double *t_gradphi; 
double *t_force; 
double *t_velocity; 



  targetInit(nSites, nFields);

  targetCalloc((void **) &t_f, nSites*nFields*sizeof(double));
  targetCalloc((void **) &t_phi, nSites*sizeof(double));
  targetCalloc((void **) &t_delsqphi, nSites*sizeof(double));
  targetCalloc((void **) &t_gradphi, nSites*3*sizeof(double));
  targetCalloc((void **) &t_force, nSites*3*sizeof(double));
  targetCalloc((void **) &t_velocity, nSites*3*sizeof(double));

  checkTargetError("Binary Collision Allocation");



  //  init_fields_target();
  

  copyToTargetMasked(t_f,f_,nSites,nFields,siteMask); 
  copyToTargetMaskedAoS(t_phi,phi_site,nSites,1,siteMask); 
  copyToTargetMaskedAoS(t_delsqphi,phi_delsq_,nSites,1,siteMask); 
  copyToTargetMaskedAoS(t_gradphi,phi_grad_,nSites,3,siteMask); 
  copyToTargetMaskedAoS(t_force,f,nSites,3,siteMask); 
  copyToTargetMaskedAoS(t_velocity,u,nSites,3,siteMask); 

  //  put_fields_on_target_masked();
  // end lattice operation setup

  //start constant setup


  copyConstantDoubleToTarget(&tc_rtau_shear, &rtau_shear, sizeof(double)); 
  copyConstantDoubleToTarget(&tc_rtau_bulk, &rtau_bulk, sizeof(double));
  copyConstantDouble1DArrayToTarget(tc_rtau, rtau_, NVEL*sizeof(double)); 
  copyConstantDouble1DArrayToTarget(tc_wv, wv, NVEL*sizeof(double));
  copyConstantDouble2DArrayToTarget( (double **) tc_ma, (double*) ma_, NVEL*NVEL*sizeof(double));
  copyConstantDouble2DArrayToTarget((double **) tc_mi, (double*) mi_, NVEL*NVEL*sizeof(double));
  copyConstantInt2DArrayToTarget((int **) tc_cv,(int*) cv, NVEL*3*sizeof(int)); 
  copyConstantDoubleToTarget(&tc_rtau2, &rtau2, sizeof(double));
  copyConstantDoubleToTarget(&tc_rcs2, &rcs2, sizeof(double));
  copyConstantIntToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstantDouble1DArrayToTarget(tc_force_global,force_global, 3*sizeof(double)); 
  copyConstantDouble2DArrayToTarget((double **) tc_d, (double*) d_, 3*3*sizeof(double));
  copyConstantDouble3DArrayToTarget((double ***) tc_q, (double *)q_, NVEL*3*3*sizeof(double)); 
  checkTargetError("constants");
  //end constant setup


  collision_binary_lb_lattice TARGET_LAUNCH(nSites) ( t_f, t_force, t_velocity,t_phi,t_gradphi,t_delsqphi,t_chemical_stress,t_chemical_potential,nSites);

  syncTarget();
  checkTargetError("Binary Collision Kernel");

  //  end lattice operation


  printf("hello2\n");
  
  //start lattice operation cleanup
    copyFromTargetMasked(f_,t_f,nSites,nFields,siteMask); 
  copyFromTargetMaskedAoS(u,t_velocity,nSites,3,siteMask); 

  //  get_fields_from_target_masked();

  targetFree(t_f);
  targetFree(t_phi);
  targetFree(t_delsqphi);
  targetFree(t_gradphi);
  targetFree(t_force);
  targetFree(t_velocity);

  targetFree(t_chemical_potential);

  checkTargetError("Binary Collision Free");
  //end lattice operation cleanup

  targetFinalize();

  return;
}
