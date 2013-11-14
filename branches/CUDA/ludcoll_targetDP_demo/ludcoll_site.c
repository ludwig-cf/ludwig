/*
 * ludcol site.c: lattice site operations for ludwig collision benchmark 
 * Alan Gray, November 2013
 */



#include <stdio.h>
#include <omp.h>
#include "ludcoll.h"


/* constant variables (declared in ludcoll.c) */
extern TARGET_CONST int N_cd[3];
extern TARGET_CONST int Nall_cd[3];
extern TARGET_CONST int nhalo_cd;
extern TARGET_CONST int nsites_cd;
extern TARGET_CONST int nop_cd;
extern TARGET_CONST double rtau_shear_d;
extern TARGET_CONST double rtau_bulk_d;
extern TARGET_CONST double rtau_d[NVEL];
extern TARGET_CONST double wv_cd[NVEL];
extern TARGET_CONST double ma_cd[NVEL][NVEL];
extern TARGET_CONST double mi_cd[NVEL][NVEL];
extern TARGET_CONST double q_cd[NVEL][3][3];
extern TARGET_CONST int cv_cd[NVEL][3];
extern TARGET_CONST double d_cd[3][3];
extern TARGET_CONST double a_d;
extern TARGET_CONST double b_d;
extern TARGET_CONST double kappa_d;
extern TARGET_CONST double rtau2_d;
extern TARGET_CONST double rcs2_d;
extern TARGET_CONST double force_global_cd[3];


/* collision operations on the lattice site */
TARGET void collision_site(
			   double* __restrict__ f_d, 
			   const double* __restrict__ ftmp_d, 
			   const double* __restrict__ phi_site_d,		
			   const double* __restrict__ grad_phi_site_d,	
			   const double* __restrict__ delsq_phi_site_d,	
			   const double* __restrict__ force_d, 
			   double* __restrict__ velocity_d, 
			   const int base_index) 
{
  int       p, m;                    /* velocity index */
  int       i, j;                    /* summed over indices ("alphabeta") */

  double    mode[NVEL*NILP];              /* Modes; hydrodynamic + ghost */
  double    rho[NILP], rrho[NILP];               /* Density, reciprocal density */
  double    u[3*NILP];                    /* Velocity */
  double    s[3][3*NILP];                 /* Stress */
  double    seq[3][3*NILP];               /* equilibrium stress */
  double    shat[3][3*NILP];              /* random stress */
  double    ghat[NVEL*NILP];              /* noise for ghosts */

  double    force[3*NILP];                /* External force */
  double    tr_s[NILP], tr_seq[NILP];

  double f_loc[2*NVEL*NILP]; /* thread local copy of f_ data */

  const double   r3     = (1.0/3.0);

  double    phi[NILP], jdotc[NILP], sphidotq[NILP];    /* modes */
  double    jphi[3*NILP];
  double    sth[3][3*NILP], sphi[3][3*NILP];
  double    mu[NILP];                      /* Chemical potential */
  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */


  int ia, ib;

  int il=0;


  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	TARGET_ILP(il) shat[ia][ib*NILP+il] = 0.0;
    }
  }
  
  for (ia = NHYDRO; ia < NVEL; ia++) {
      TARGET_ILP(il) ghat[ia*NILP+il] = 0.0;
  }
  


  /* load data */
  for(p = 0; p < NVEL; p++) {
    for(m = 0; m < NDIST; m++) {
      TARGET_ILP(il) f_loc[(NVEL*m+p)*NILP+il] = ftmp_d[nsites_cd*NDIST*p + nsites_cd*m + base_index+il];
    }
  }

  /* matrix multiplication for full SIMD vector */
  for (m = 0; m < NVEL; m++) {
      TARGET_ILP(il) mode[m*NILP+il] = 0.0;
    for (p = 0; p < NVEL; p++) {
      TARGET_ILP(il) mode[m*NILP+il] += f_loc[p*NILP+il]*ma_cd[m][p];
    }
    
  }

  /* loop over SIMD vector of lattice sites */
  
      /* For convenience, write out the physical modes. */
      
      TARGET_ILP(il) rho[il] = mode[0+il];
      for (i = 0; i < 3; i++) {
	TARGET_ILP(il) u[i*NILP+il] = mode[(1 + i)*NILP+il];
      }

      TARGET_ILP(il)
	{
     

	  s[X][X*NILP+il] = mode[4*NILP+il];
	  s[X][Y*NILP+il] = mode[5*NILP+il];
	  s[X][Z*NILP+il] = mode[6*NILP+il];
	  s[Y][X*NILP+il] = s[X][Y*NILP+il];
	  s[Y][Y*NILP+il] = mode[7*NILP+il];
	  s[Y][Z*NILP+il] = mode[8*NILP+il];
	  s[Z][X*NILP+il] = s[X][Z*NILP+il];
	  s[Z][Y*NILP+il] = s[Y][Z*NILP+il];
	  s[Z][Z*NILP+il] = mode[9*NILP+il];
      
	}


  

  /* Compute the local velocity, taking account of any body force */
  TARGET_ILP(il)  rrho[il] = 1.0/rho[il];
  
  for (i = 0; i < 3; i++) {
    TARGET_ILP(il)
      {
      force[i*NILP+il] = (force_global_cd[i] + force_d[i*nsites_cd+base_index+il]);
      u[i*NILP+il] = rrho[il]*(u[i*NILP+il] + 0.5*force[i*NILP+il]);
      }
  }
  /* hydrodynamics_set_velocity(index, u); */
  for (i = 0; i < 3; i++) {
    TARGET_ILP(il)
      velocity_d[i*nsites_cd+base_index+il] = u[i*NILP+il];
  }
  
  /* Compute the thermodynamic component of the stress */
  
  
  double p0[NILP];

  TARGET_ILP(il)
    {

  p0[il] = 0.5*a_d*phi_site_d[base_index+il]*phi_site_d[base_index+il] 
    + 0.75*b_d*phi_site_d[base_index+il]*phi_site_d[base_index+il]
    *phi_site_d[base_index+il]*phi_site_d[base_index+il]
    - kappa_d*phi_site_d[base_index+il]*delsq_phi_site_d[base_index+il] - 
    0.5*kappa_d*
    (grad_phi_site_d[X*nsites_cd+base_index+il]*grad_phi_site_d[X*nsites_cd+base_index+il] 
     + grad_phi_site_d[Y*nsites_cd+base_index+il]*grad_phi_site_d[Y*nsites_cd+base_index+il] 
     + grad_phi_site_d[Z*nsites_cd+base_index+il]*grad_phi_site_d[Z*nsites_cd+base_index+il]);
    }


  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	
      TARGET_ILP(il) sth[ia][ib*NILP+il] = p0[il]*d_cd[ia][ib]	+ kappa_d*grad_phi_site_d[ia*nsites_cd+base_index+il]*grad_phi_site_d[ib*nsites_cd+base_index+il];

    }
  }





  /* Relax stress with different shear and bulk viscosity */
  TARGET_ILP(il)
  {  
  tr_s[il]   = 0.0;
  tr_seq[il] = 0.0;
  }
  

  for (i = 0; i < 3; i++) {
    /* Set equilibrium stress, which includes thermodynamic part */
    for (j = 0; j < 3; j++) {
      TARGET_ILP(il)
	seq[i][j*NILP+il] = rho[il]*u[i*NILP+il]*u[j*NILP+il] + sth[i][j*NILP+il];
    }
    /* Compute trace */

    TARGET_ILP(il) 
      {
	tr_s[il]   += s[i][i*NILP+il];
	tr_seq[il] += seq[i][i*NILP+il];
      }
  }
  

  /* Form traceless parts */
  for (i = 0; i < 3; i++) {
    TARGET_ILP(il) 
      {
	s[i][i*NILP+il]   -= r3*tr_s[il];
	seq[i][i*NILP+il] -= r3*tr_seq[il];
      }

  }
  
  /* Relax each mode */
  TARGET_ILP(il)
    tr_s[il] = tr_s[il] - rtau_bulk_d*(tr_s[il] - tr_seq[il]);
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {

	TARGET_ILP(il) 
	  {

	    s[i][j*NILP+il] -= rtau_shear_d*(s[i][j*NILP+il] - seq[i][j*NILP+il]);
	    s[i][j*NILP+il] += d_cd[i][j]*r3*tr_s[il];
	
	    /* Correction from body force (assumes equal relaxation times) */
	    
	    s[i][j*NILP+il] += (2.0-rtau_shear_d)*(u[i*NILP+il]*force[j*NILP+il] + force[i*NILP+il]*u[j*NILP+il]);
	    shat[i][j*NILP+il] = 0.0;
      }
    }
  }
  
  //if (isothermal_fluctuations_) fluctuations_on(shat, ghat);
  
  /* Now reset the hydrodynamic modes to post-collision values */


  TARGET_ILP(il)
    {
  
      mode[1*NILP+il] = mode[1*NILP+il] + force[X*NILP+il];    /* Conserved if no force */
      mode[2*NILP+il] = mode[2*NILP+il] + force[Y*NILP+il];    /* Conserved if no force */
      mode[3*NILP+il] = mode[3*NILP+il] + force[Z*NILP+il];    /* Conserved if no force */
      mode[4*NILP+il] = s[X][X*NILP+il] + shat[X][X*NILP+il];
      mode[5*NILP+il] = s[X][Y*NILP+il] + shat[X][Y*NILP+il];
      mode[6*NILP+il] = s[X][Z*NILP+il] + shat[X][Z*NILP+il];
      mode[7*NILP+il] = s[Y][Y*NILP+il] + shat[Y][Y*NILP+il];
      mode[8*NILP+il] = s[Y][Z*NILP+il] + shat[Y][Z*NILP+il];
      mode[9*NILP+il] = s[Z][Z*NILP+il] + shat[Z][Z*NILP+il];
    }  
  /* Ghost modes are relaxed toward zero equilibrium. */
  
  for (m = NHYDRO; m < NVEL; m++) {
    TARGET_ILP(il) mode[m*NILP+il] = mode[m*NILP+il] - rtau_d[m]*(mode[m*NILP+il] - 0.0) + ghat[m*NILP+il];
  }



  double ftmp[NILP];

  /* Project post-collision modes back onto the distribution */
  /* matrix multiplication for full SIMD vector */
  for (p = 0; p < NVEL; p++) {
    TARGET_ILP(il) ftmp[il] = 0.;
    for (m = 0; m < NVEL; m++) {
  	TARGET_ILP(il) ftmp[il] += mi_cd[p][m]*mode[m*NILP+il];
    }
      TARGET_ILP(il) f_d[nsites_cd*NDIST*p + base_index+il]=ftmp[il];
  }
     

  /* Now, the order parameter distribution */
 
  TARGET_ILP(il)
    {
 
    phi[il] = phi_site_d[base_index+il];

    mu[il] = a_d*phi[il] + b_d*phi[il]*phi[il]*phi[il] - kappa_d*delsq_phi_site_d[base_index+il];
  
    jphi[X*NILP+il] = 0.0;
    jphi[Y*NILP+il] = 0.0;
    jphi[Z*NILP+il] = 0.0;
  }
  for (p = 1; p < NVEL; p++) {
    for (i = 0; i < 3; i++) {
      //jphi[i] += f_loc[NVEL + p]*cv_cd[p][i];
	TARGET_ILP(il) jphi[i*NILP+il] += f_loc[(NVEL + p)*NILP+il]*cv_cd[p][i];
    }
  }


  /* Relax order parameters modes. See the comments above. */
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      TARGET_ILP(il) sphi[i][j*NILP+il] = phi[il]*u[i*NILP+il]*u[j*NILP+il] 
	+ mu[il]*d_cd[i][j];
    }
    TARGET_ILP(il) jphi[i*NILP+il] = jphi[i*NILP+il] - rtau2_d*(jphi[i*NILP+il] 
				    - phi[il]*u[i*NILP+il]);
  }
  

  /* Now update the distribution */
	
  for (p = 0; p < NVEL; p++) {

  int dp0 = (p == 0);

  TARGET_ILP(il) 
    {     
      jdotc[il]    = 0.0;
      sphidotq[il] = 0.0;
    }    

    for (i = 0; i < 3; i++) {
	TARGET_ILP(il) jdotc[il] += jphi[i*NILP+il]*cv_cd[p][i];
      for (j = 0; j < 3; j++) {
	TARGET_ILP(il) sphidotq[il] += sphi[i][j*NILP+il]*q_cd[p][i][j];
      }
    }
    
    /* Project all this back to the distributions. The magic
     * here is to move phi into the non-propagating distribution. */
    
    TARGET_ILP(il) f_d[nsites_cd*NDIST*p+nsites_cd+base_index+il]
      = wv_cd[p]*(jdotc[il]*rcs2_d + sphidotq[il]*r2rcs4) + phi[il]*dp0;
    


  }
    
  


  return;
}






// version with no instruction level parallelism for timing comparison
TARGET void collision_site_NOILP(
				 double* __restrict__ f_d, 
				 const double* __restrict__ ftmp_d, 
				 const double* __restrict__ phi_site_d,		
				 const double* __restrict__ grad_phi_site_d,	
				 const double* __restrict__ delsq_phi_site_d,	
				 const double* __restrict__ force_d, 
				 double* __restrict__ velocity_d, const int index) 
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

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      shat[ia][ib] = 0.0;
    }
  }

  for (ia = NHYDRO; ia < NVEL; ia++) {
    ghat[ia] = 0.0;
  }


  /* load data */
  for(p = 0; p < NVEL; p++) {
    for(m = 0; m < NDIST; m++) {
      f_loc[NVEL*m+p] = ftmp_d[nsites_cd*NDIST*p + nsites_cd*m + index];
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
    force_local[i] = force_d[i*nsites_cd+index];
  }
  
  for (i = 0; i < 3; i++) {
    force[i] = (force_global_cd[i] + force_local[i]);
    u[i] = rrho*(u[i] + 0.5*force[i]);
  }
  /* hydrodynamics_set_velocity(index, u); */
  for (i = 0; i < 3; i++) {
   velocity_d[i*nsites_cd+index] = u[i];
  }
  

  /* Compute the thermodynamic component of the stress */
  
  
  double p0;

  p0 = 0.5*a_d*phi_site_d[index]*phi_site_d[index] 
    + 0.75*b_d*phi_site_d[index]*phi_site_d[index]
    *phi_site_d[index]*phi_site_d[index]
    - kappa_d*phi_site_d[index]*delsq_phi_site_d[index] - 
    0.5*kappa_d*
    (grad_phi_site_d[X*nsites_cd+index]*grad_phi_site_d[X*nsites_cd+index] 
     + grad_phi_site_d[Y*nsites_cd+index]*grad_phi_site_d[Y*nsites_cd+index] 
     + grad_phi_site_d[Z*nsites_cd+index]*grad_phi_site_d[Z*nsites_cd+index]);
    


  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	
      sth[ia][ib] = p0*d_cd[ia][ib]	+ kappa_d*grad_phi_site_d[ia*nsites_cd+index]*grad_phi_site_d[ib*nsites_cd+index];

    }
  }
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
    f_d[nsites_cd*NDIST*p + index] = f_tmp;
  }
  
  /* Now, the order parameter distribution */
  
  phi = phi_site_d[index];

  mu = a_d*phi + b_d*phi*phi*phi - kappa_d*delsq_phi_site_d[index];
  
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
    
    f_d[nsites_cd*NDIST*p+nsites_cd+index]
      = wv_cd[p]*(jdotc*rcs2_d + sphidotq*r2rcs4) + phi*dp0;
    
  }
  
  


  return;
}






// version with no instruction level parallelism for timing comparison
// HACKED TRANSPOSED data structures for optimal memory access
// THIS VERSION GIVES INCORRECT RESULTS BUT INDICATES TIMINGS
TARGET void collision_site_NOILP_TRANSPOSE(
				 double* __restrict__ f_d, 
				 const double* __restrict__ ftmp_d, 
				 const double* __restrict__ phi_site_d,		
				 const double* __restrict__ grad_phi_site_d,	
				 const double* __restrict__ delsq_phi_site_d,	
				 const double* __restrict__ force_d, 
				 double* __restrict__ velocity_d, const int index) 
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

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      shat[ia][ib] = 0.0;
    }
  }

  for (ia = NHYDRO; ia < NVEL; ia++) {
    ghat[ia] = 0.0;
  }


  /* load data */

    for(m = 0; m < NDIST; m++) {
      for(p = 0; p < NVEL; p++) {
      //f_loc[NVEL*m+p] = ftmp_d[nsites_cd*NDIST*p + nsites_cd*m + index];
	f_loc[NVEL*m+p] = ftmp_d[index*NVEL*NDIST+NVEL*m+p];
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
    //force_local[i] = force_d[i*nsites_cd+index];
    force_local[i] = force_d[index*3+i];
  }
  
  for (i = 0; i < 3; i++) {
    force[i] = (force_global_cd[i] + force_local[i]);
    u[i] = rrho*(u[i] + 0.5*force[i]);
  }
  /* hydrodynamics_set_velocity(index, u); */
  for (i = 0; i < 3; i++) {
    //velocity_d[i*nsites_cd+index] = u[i];
    velocity_d[index*3+i] = u[i];
  }
  

  /* Compute the thermodynamic component of the stress */
  
  
  double p0;

  p0 = 0.5*a_d*phi_site_d[index]*phi_site_d[index] 
    + 0.75*b_d*phi_site_d[index]*phi_site_d[index]
    *phi_site_d[index]*phi_site_d[index]
    - kappa_d*phi_site_d[index]*delsq_phi_site_d[index] - 
    0.5*kappa_d*
    //    (grad_phi_site_d[X*nsites_cd+index]*grad_phi_site_d[X*nsites_cd+index] 
    //     + grad_phi_site_d[Y*nsites_cd+index]*grad_phi_site_d[Y*nsites_cd+index] 
    //     + grad_phi_site_d[Z*nsites_cd+index]*grad_phi_site_d[Z*nsites_cd+index]);

    (grad_phi_site_d[index*3+X]*grad_phi_site_d[index*3+X] 
     + grad_phi_site_d[index*3+Y]*grad_phi_site_d[index*3+Y] 
     + grad_phi_site_d[index*3+Z]*grad_phi_site_d[index*3+X]);
    


  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
	
      //     sth[ia][ib] = p0*d_cd[ia][ib]	+ kappa_d*grad_phi_site_d[ia*nsites_cd+index]*grad_phi_site_d[ib*nsites_cd+index];
      sth[ia][ib] = p0*d_cd[ia][ib]	+ kappa_d*grad_phi_site_d[index*3+ia]*grad_phi_site_d[index*3+ib];

    }
  }
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
    //f_d[nsites_cd*NDIST*p + index] = f_tmp;
    f_d[index*NDIST*NVEL+p] = f_tmp;
  }
  
  /* Now, the order parameter distribution */
  
  phi = phi_site_d[index];

  mu = a_d*phi + b_d*phi*phi*phi - kappa_d*delsq_phi_site_d[index];
  
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
    
    //f_d[nsites_cd*NDIST*p+nsites_cd+index]
    //= wv_cd[p]*(jdotc*rcs2_d + sphidotq*r2rcs4) + phi*dp0;

    f_d[index*NVEL*NDIST+NVEL+p]
      = wv_cd[p]*(jdotc*rcs2_d + sphidotq*r2rcs4) + phi*dp0;


    
  }
  
  


  return;
}
