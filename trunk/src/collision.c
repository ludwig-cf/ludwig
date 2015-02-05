/*****************************************************************************
 *
 *  collision.c
 *
 *  Collision stage routines and associated data.
 *
 *  Isothermal fluctuations following Adhikari et al., Europhys. Lett
 *  (2005).
 *
 *  The relaxation times can be set to give either 'm10', BGK or
 *  'two-relaxation' time (TRT) models.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "model.h"
#include "lb_model_s.h"
#include "hydro_s.h"
#include "free_energy.h"
#include "control.h"
#include "collision.h"
#include "field_s.h"



static int nmodes_ = NVEL;               /* Modes to use in collsion stage */
static int nrelax_ = RELAXATION_M10;     /* [RELAXATION_M10|TRT|BGK] */
                                         /* Default is M10 */

static double rtau_shear;       /* Inverse relaxation time for shear modes */
static double rtau_bulk;        /* Inverse relaxation time for bulk modes */
static double var_shear;        /* Variance for shear mode fluctuations */
static double var_bulk;         /* Variance for bulk mode fluctuations */
static double rtau_[NVEL];      /* Inverse relaxation times */
static double noise_var[NVEL];  /* Noise variances */

static int lb_collision_mrt(lb_t * lb, hydro_t * hydro, map_t * map, noise_t * noise);
static int lb_collision_binary(lb_t * lb, hydro_t * hydro, map_t * map, noise_t * noise);
HOST TARGET static int fluctuations_off(double shat[3][3], double ghat[NVEL]);
static int collision_fluctuations(noise_t * noise, int index,
				  double shat[3][3], double ghat[NVEL]);


//TODO refactor these type definitions and forward declarations

typedef double (*mu_fntype)(const int, const int, const double*, const double*);
typedef void (*pth_fntype)(const int, double(*)[3*NILP], const double*, const double*, const double*);

HOST void get_chemical_stress_target(pth_fntype* t_chemical_stress);
HOST void get_chemical_potential_target(mu_fntype* t_chemical_potential);
HOST void symmetric_phi(double** address_of_ptr);
HOST void symmetric_gradphi(double** address_of_ptr);
HOST void symmetric_delsqphi(double** address_of_ptr);
HOST void symmetric_t_phi(double** address_of_ptr);
HOST void symmetric_t_gradphi(double** address_of_ptr);
HOST void symmetric_t_delsqphi(double** address_of_ptr);
HOST char symmetric_in_use();

/*****************************************************************************
 *
 *  lb_collide
 *
 *  Driver routine for the collision stage.
 *
 *  We allow hydro to be NULL, in which case there is no hydrodynamics!
 * 
 *****************************************************************************/

int lb_collide(lb_t * lb, hydro_t * hydro, map_t * map, noise_t * noise) {

  int ndist;

  if (hydro == NULL) return 0;

  assert(lb);
  assert(map);

  lb_ndist(lb, &ndist);
  lb_collision_relaxation_times_set(noise);

  if (ndist == 1) lb_collision_mrt(lb, hydro, map, noise);
  if (ndist == 2) lb_collision_binary(lb, hydro, map, noise);

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_mrt
 *
 *  Collision with (potentially) different relaxation times for each
 *  different mode.
 *
 *  The matrices ma_ and mi_ project the distributions onto the
 *  modes, and vice-versa, respectively, for the current LB model.
 *
 *  The collision conserves density, and momentum (to within any
 *  body force present). The stress modes, and ghost modes, are
 *  relaxed toward their equilibrium values.
 *
 *  If ghost modes are not required, nmodes_ can be set equal to
 *  the number of hydrodynamic modes. Otherwise nmodes_ = NVEL.  
 *
 *****************************************************************************/

int lb_collision_mrt(lb_t * lb, hydro_t * hydro, map_t * map, noise_t * noise) {

  int       nlocal[3];
  int       ic, jc, kc, index;       /* site indices */
  int       p, m;                    /* velocity index */
  int       ia, ib;                  /* indices ("alphabeta") */
  int       status;
  int       noise_on = 0;

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
  double    force_global[3];

  /* SIMD stuff */
  int iv,base_index;
  int nv,full_vec;
  
  double f_v[NVEL][SIMDVL];
  double mode_v[NVEL][SIMDVL];

  assert(lb);
  assert(hydro);
  assert(map);

  coords_nlocal(nlocal);
  fluctuations_off(shat, ghat);
  physics_fbody(force_global);

  rdim = 1.0/NDIM;

  for (ia = 0; ia < 3; ia++) {
    u[ia] = 0.0;
  }

  noise_present(noise, NOISE_RHO, &noise_on);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      /* loop over Z index in steps of size SIMDVL */

      for (kc = 1; kc <= nlocal[Z]; kc += SIMDVL) {
	
	/* nv is the number of SIMD iterations: i.e. SIMDVL unless
	 * this overflows the dimension, in which it becomes the number of
	 * remaining valid sites. Note that, we need to use the CPP variable
	 * SIMDVL rather than the runtime nv where possible in key loops to
	 * help compiler optimisation */

	nv = SIMDVL;
	full_vec = 1;
	if ( kc > nlocal[Z] - SIMDVL + 1 ) {
	  full_vec = 0;
	  nv = nlocal[Z] + 1 - kc;
	}
	
	base_index = coords_index(ic, jc, kc);	
	
	/* Compute all the modes */	
	/* load SIMD vector of lattice sites */

	if ( full_vec ) {
	  lb_f_multi_index(lb, base_index, 0, f_v);
	}
	else {
	  lb_f_multi_index_part(lb, base_index, 0, f_v, nv);
	}

	/* matrix multiplication for full SIMD vector */

	for (m = 0; m < nmodes_; m++) {
	  for (iv = 0; iv < SIMDVL; iv++)
	    mode_v[m][iv] = 0.0;
	  for (p = 0; p < NVEL; p++) {
	    for (iv = 0; iv < SIMDVL; iv++) {
	      mode_v[m][iv] += f_v[p][iv]*ma_[m][p];
	    }
	  }
	  
	}
	
	/* loop over SIMD vector of lattice sites */

	for (iv = 0; iv < nv; iv++) {
	  
	  index = base_index + iv;
	  map_status(map, index, &status);
	  if (status != MAP_FLUID) continue;

	  for (m = 0; m < nmodes_; m++) { 
	    mode[m] = mode_v[m][iv];
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
	  hydro_f_local(hydro, index, force_local);
	  
	  for (ia = 0; ia < NDIM; ia++) {
	    force[ia] = (force_global[ia] + force_local[ia]);
	    u[ia] = rrho*(u[ia] + 0.5*force[ia]);
	  }
	  hydro_u_set(hydro, index, u);
	  
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
	  tr_s = tr_s - rtau_bulk*(tr_s - tr_seq);
	  
	  for (ia = 0; ia < NDIM; ia++) {
	    for (ib = 0; ib < NDIM; ib++) {
	      s[ia][ib] -= rtau_shear*(s[ia][ib] - seq[ia][ib]);
	      s[ia][ib] += d_[ia][ib]*rdim*tr_s;
	      
	      /* Correction from body force (assumes equal relaxation times) */
	      
	      s[ia][ib] += (2.0-rtau_shear)*(u[ia]*force[ib] + force[ia]*u[ib]);
	    }
	  }
	  
	  if (noise_on) collision_fluctuations(noise, index, shat, ghat);
	  
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
	  
	  for (m = NHYDRO; m < nmodes_; m++) {
	    mode[m] = mode[m] - rtau_[m]*(mode[m] - 0.0) + ghat[m];
	  }
	  
	  for (m = 0; m < nmodes_; m++) {
	    mode_v[m][iv] = mode[m];
	  }
	  
	  /* next SIMD vector */
	}
	
	/* Project post-collision modes back onto the distribution */
	/* matrix multiplication for full SIMD vector */

	for (p = 0; p < NVEL; p++) {
	  for (iv = 0; iv < SIMDVL; iv++) 
	    f_v[p][iv] = 0.0;
	  for (m = 0; m < nmodes_; m++) {
	    for (iv = 0; iv < SIMDVL; iv++) 
	      f_v[p][iv] += mi_[p][m]*mode_v[m][iv];
	  }
	}
	
	/* Store SIMD vector of lattice sites */

	if ( full_vec ) {
	  lb_f_multi_index_set(lb, base_index, 0, f_v);
	}
	else {
	  lb_f_multi_index_set_part(lb, base_index, 0, f_v, nv);
	}
	
	/* Next site */
      }
    }
  }
  
  return 0;
}


// binary collision code below has been ported to targetDP programming model.

/* Constants*/

TARGET_CONST int tc_nSites;
TARGET_CONST double tc_rtau_shear;
TARGET_CONST double tc_rtau_bulk;
TARGET_CONST double tc_rtau_[NVEL];
TARGET_CONST double tc_wv[NVEL];
TARGET_CONST double tc_ma_[NVEL][NVEL];
TARGET_CONST double tc_mi_[NVEL][NVEL];
TARGET_CONST int tc_cv[NVEL][3];
TARGET_CONST double tc_rtau2;
TARGET_CONST double tc_rcs2;
TARGET_CONST double tc_r2rcs4;
TARGET_CONST double tc_force_global[3];
TARGET_CONST double tc_q_[NVEL][3][3];
TARGET_CONST int tc_nmodes_; 

// target copies of fields 
static double *t_phi; 
static double *t_delsqphi; 
static double *t_gradphi; 

#define NDIST 2 //for binary collision


/*****************************************************************************
 *
 *  lb_collision_binary
 *
 *  Binary LB collision stage (here we are progressing toward
 *  decoupled version).
 *
 *  This follows the single fluid version above, with the addition
 *  that the equilibrium stress includes the thermodynamic term
 *  following Swift etal.
 *
 *  We also have to update the second distribution g from the
 *  order parameter modes phi, jphi[3], sphi[3][3].
 *
 *  There are two choices:
 *    1. relax jphi[i] toward equilibrium phi*u[i] at rate rtau2
 *       AND
 *       fix sphi[i][j] = phi*u[i]*u[j] + mu*d_[i][j]
 *       so the mobility enters through rtau2 (J. Stat. Phys. 2005).
 *    2.
 *       fix jphi[i] = phi*u[i] (i.e. relaxation time == 1.0)
 *       AND
 *       fix sphi[i][j] = phi*u[i]*u[j] + mobility*mu*d_[i][j]
 *       so the mobility enters with chemical potential (Kendon etal 2001).
 *
 *       Note thare is an extra factor of cs^2 before the mobility,
 *       which should be taken into account if quoting the actual
 *       mobility. The factor is somewhat arbitrary and is there
 *       to try to ensure both methods are stable for given input
 *       mobility.
 *
 *   As there seems to be little to choose between the two in terms of
 *   results, I prefer 2, as it avoids the calculation of jphi[i] from
 *   from the distributions g. However, keep 1 so tests don't break!
 *
 *   However, for asymmetric quenches version 1 may be preferred.
 *
 *   The reprojection of g moves phi (mostly) into the non-propagating
 *   distribution following J. Stat. Phys. (2005).
 *
 *****************************************************************************/


// first we define the function applied to each lattice site
TARGET void lb_collision_binary_site( double* __restrict__ t_f, 
			      const double* __restrict__ t_force, 
			      double* __restrict__ t_velocity,
			      double* __restrict__ t_phi,
			      double* __restrict__ t_gradphi,
			      double* __restrict__ t_delsqphi,
			      pth_fntype* t_chemical_stress,
			      mu_fntype* t_chemical_potential, 
			       noise_t * noise, int noise_on,
			      const int baseIndex){


  int i,j,m,p;



  //TARGETDP SYNTAX GUIDE:
  //DECLARE_SIMD_* macros are used to declare automatic data structures 
  //where each holds a vector "chunk" of lattice sites to allow SIMD operations.
  //e.g. "DECLARE_SIMD_VECTOR1D(type, var, size, iv) is equivalent of "type var[size]", where the extra index iv 
  // will automatically index into the short SIMD array when used in conjuction with TARGET_ILP.  
  
  DECLARE_SIMD_VECTOR1D(double, mode, NVEL); /* Modes; hydrodynamic + ghost */

  /* Density, reciprocal density */
  DECLARE_SIMD_SCALAR(double,rho);
  DECLARE_SIMD_SCALAR(double,rrho);
    
  DECLARE_SIMD_VECTOR1D(double, u, 3);       /* Velocity */
  DECLARE_SIMD_VECTOR2D(double,s,3,3);       /* Stress */
  DECLARE_SIMD_VECTOR2D(double,seq,3,3);     /* equilibrium stress */
  DECLARE_SIMD_VECTOR2D(double,shat,3,3);    /* random stress */
  DECLARE_SIMD_VECTOR1D(double, ghat, NVEL); /* noise for ghosts */
  
  DECLARE_SIMD_VECTOR1D(double, force, 3);  /* External force */

  DECLARE_SIMD_SCALAR(double,tr_s);
  DECLARE_SIMD_SCALAR(double,tr_seq);
  
  /* modes */
  DECLARE_SIMD_SCALAR(double,phi);
  DECLARE_SIMD_SCALAR(double,jdotc);
  DECLARE_SIMD_SCALAR(double,sphidotq);    

  DECLARE_SIMD_VECTOR1D(double, jphi, 3); 

  DECLARE_SIMD_VECTOR2D(double,sth,3,3);
  DECLARE_SIMD_VECTOR2D(double,sphi,3,3);

  DECLARE_SIMD_SCALAR(double,mu);    /* Chemical potential */
  

  /* index for SIMD vectors */
  int iv=0;        

  /* switch fluctuations off */
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      TARGET_ILP(iv) SIMD_2D_ELMNT(shat,i,j,iv) = 0.0;
    }
  }

  for (i = NHYDRO; i < NVEL; i++) {
    TARGET_ILP(iv) SIMD_1D_ELMNT(ghat,i,iv) = 0.0;
  }


  /* Compute all the modes */
  for (m = 0; m < tc_nmodes_; m++) {
    TARGET_ILP(iv) SIMD_1D_ELMNT(mode,m,iv) = 0.0;
    for (p = 0; p < NVEL; p++) {
      TARGET_ILP(iv) SIMD_1D_ELMNT(mode,m,iv) +=
  	t_f[ LB_ADDR(tc_nSites, NDIST, NVEL, baseIndex + iv, 0, p) ]
  	*tc_ma_[m][p];
    }
    
  }


  
  //map status is now taken care of in masked targetDP data copies.
  // we perform calculations for all sites, and only copy back the fluid
  //sites to the host
  //int status;
  //map_status(map, baseIndex, &status);
  //if (status != MAP_FLUID) return;
  
  /* For convenience, write out the physical modes. */
  
  TARGET_ILP(iv) SIMD_SC_ELMNT(rho,iv) = SIMD_1D_ELMNT(mode,0,iv);
  for (i = 0; i < 3; i++) {
    TARGET_ILP(iv) SIMD_1D_ELMNT(u,i,iv) = SIMD_1D_ELMNT(mode,1 + i,iv);
  }
  
  TARGET_ILP(iv) {
    SIMD_2D_ELMNT(s,X,X,iv) = SIMD_1D_ELMNT(mode,4,iv);
    SIMD_2D_ELMNT(s,X,Y,iv) = SIMD_1D_ELMNT(mode,5,iv);
    SIMD_2D_ELMNT(s,X,Z,iv) = SIMD_1D_ELMNT(mode,6,iv);
    SIMD_2D_ELMNT(s,Y,X,iv) = SIMD_2D_ELMNT(s,X,Y,iv);
    SIMD_2D_ELMNT(s,Y,Y,iv) = SIMD_1D_ELMNT(mode,7,iv);
    SIMD_2D_ELMNT(s,Y,Z,iv) = SIMD_1D_ELMNT(mode,8,iv);
    SIMD_2D_ELMNT(s,Z,X,iv) = SIMD_2D_ELMNT(s,X,Z,iv);
    SIMD_2D_ELMNT(s,Z,Y,iv) = SIMD_2D_ELMNT(s,Y,Z,iv);
    SIMD_2D_ELMNT(s,Z,Z,iv) = SIMD_1D_ELMNT(mode,9,iv);
  }

  /* Compute the local velocity, taking account of any body force */
  
  TARGET_ILP(iv) SIMD_SC_ELMNT(rrho,iv) 
    = 1.0/SIMD_SC_ELMNT(rho,iv);


  
  for (i = 0; i < 3; i++) {

    TARGET_ILP(iv){
      SIMD_1D_ELMNT(force,i,iv) = (tc_force_global[i] 
		      + t_force[HYADR(tc_nSites,3,baseIndex+iv,i)]);
      

      SIMD_1D_ELMNT(u,i,iv) = SIMD_SC_ELMNT(rrho,iv)*(SIMD_1D_ELMNT(u,i,iv) + 0.5*SIMD_1D_ELMNT(force,i,iv));  
    }
  }
  
  
    for (i = 0; i < 3; i++) {   
              TARGET_ILP(iv) t_velocity[HYADR(tc_nSites,3,baseIndex+iv,i)]=SIMD_1D_ELMNT(u,i,iv);

   }

  
  /* Compute the thermodynamic component of the stress */
  
  (*t_chemical_stress)(baseIndex, sth,  t_phi, t_gradphi, t_delsqphi);
  
  /* Relax stress with different shear and bulk viscosity */
  
  TARGET_ILP(iv){
    SIMD_SC_ELMNT(tr_s,iv)   = 0.0;
    SIMD_SC_ELMNT(tr_seq,iv) = 0.0;
  }
  
  for (i = 0; i < 3; i++) {
    /* Set equilibrium stress, which includes thermodynamic part */
    for (j = 0; j < 3; j++) {
      TARGET_ILP(iv) SIMD_2D_ELMNT(seq,i,j,iv) = SIMD_SC_ELMNT(rho,iv)*SIMD_1D_ELMNT(u,i,iv)*SIMD_1D_ELMNT(u,j,iv) 
	+ SIMD_2D_ELMNT(sth,i,j,iv);
    }
    /* Compute trace */
    TARGET_ILP(iv) {
      SIMD_SC_ELMNT(tr_s,iv)   += SIMD_2D_ELMNT(s,i,i,iv);
      SIMD_SC_ELMNT(tr_seq,iv) += SIMD_2D_ELMNT(seq,i,i,iv);
    }
  }
  
  /* Form traceless parts */
  for (i = 0; i < 3; i++) {
    TARGET_ILP(iv) {
      SIMD_2D_ELMNT(s,i,i,iv)   -= tc_r3_*SIMD_SC_ELMNT(tr_s,iv);
      SIMD_2D_ELMNT(seq,i,i,iv) -= tc_r3_*SIMD_SC_ELMNT(tr_seq,iv);
    }
  }

  
  /* Relax each mode */
  TARGET_ILP(iv)
    SIMD_SC_ELMNT(tr_s,iv) = SIMD_SC_ELMNT(tr_s,iv) - tc_rtau_bulk*(SIMD_SC_ELMNT(tr_s,iv) - SIMD_SC_ELMNT(tr_seq,iv));
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {

      TARGET_ILP(iv) {
	SIMD_2D_ELMNT(s,i,j,iv) -= tc_rtau_shear*(SIMD_2D_ELMNT(s,i,j,iv) - SIMD_2D_ELMNT(seq,i,j,iv));
	SIMD_2D_ELMNT(s,i,j,iv) += tc_d_[i][j]*tc_r3_*SIMD_SC_ELMNT(tr_s,iv);
      
	/* Correction from body force (assumes equal relaxation times) */
      
	SIMD_2D_ELMNT(s,i,j,iv) += (2.0-tc_rtau_shear)*(SIMD_1D_ELMNT(u,i,iv)*SIMD_1D_ELMNT(force,j,iv) 
					   + SIMD_1D_ELMNT(force,i,iv)*SIMD_1D_ELMNT(u,j,iv));
	SIMD_2D_ELMNT(shat,i,j,iv) = 0.0;
      }
    }
  }
  


  if (noise_on) {
    
#ifdef CUDA 
    
    printf("Error: noise_on is not yet supported for CUDA\n");
    //exit(1);
    
#else      
    
     TARGET_ILP(iv){
      
      double shattmp[3][3];
      double ghattmp[NVEL];
      
      collision_fluctuations(noise, baseIndex+iv, shattmp, ghattmp);      

      for(i=0;i<3;i++)
	for(j=0;j<3;j++)
	  SIMD_2D_ELMNT(shat,i,j,iv)=shattmp[i][j];

      for(i=0;i<NVEL;i++)
	SIMD_1D_ELMNT(ghat,i,iv)=ghattmp[i];
      

    }    

#endif
    
  }    
  
  /* Now reset the hydrodynamic modes to post-collision values */
  
  TARGET_ILP(iv) {
    SIMD_1D_ELMNT(mode,1,iv) = SIMD_1D_ELMNT(mode,1,iv) + SIMD_1D_ELMNT(force,X,iv);    /* Conserved if no force */
    SIMD_1D_ELMNT(mode,2,iv) = SIMD_1D_ELMNT(mode,2,iv) + SIMD_1D_ELMNT(force,Y,iv);    /* Conserved if no force */
    SIMD_1D_ELMNT(mode,3,iv) = SIMD_1D_ELMNT(mode,3,iv) + SIMD_1D_ELMNT(force,Z,iv);    /* Conserved if no force */
    SIMD_1D_ELMNT(mode,4,iv) = SIMD_2D_ELMNT(s,X,X,iv) + SIMD_2D_ELMNT(shat,X,X,iv);
    SIMD_1D_ELMNT(mode,5,iv) = SIMD_2D_ELMNT(s,X,Y,iv) + SIMD_2D_ELMNT(shat,X,Y,iv);
    SIMD_1D_ELMNT(mode,6,iv) = SIMD_2D_ELMNT(s,X,Z,iv) + SIMD_2D_ELMNT(shat,X,Z,iv);
    SIMD_1D_ELMNT(mode,7,iv) = SIMD_2D_ELMNT(s,Y,Y,iv) + SIMD_2D_ELMNT(shat,Y,Y,iv);
    SIMD_1D_ELMNT(mode,8,iv) = SIMD_2D_ELMNT(s,Y,Z,iv) + SIMD_2D_ELMNT(shat,Y,Z,iv);
    SIMD_1D_ELMNT(mode,9,iv) = SIMD_2D_ELMNT(s,Z,Z,iv) + SIMD_2D_ELMNT(shat,Z,Z,iv);
  }
  
  
  /* Ghost modes are relaxed toward zero equilibrium. */
  
  for (m = NHYDRO; m < tc_nmodes_; m++) {
         TARGET_ILP(iv)  SIMD_1D_ELMNT(mode,m,iv) = SIMD_1D_ELMNT(mode,m,iv) 
	   - tc_rtau_[m]*(SIMD_1D_ELMNT(mode,m,iv) - 0.0) + SIMD_1D_ELMNT(ghat,m,iv);
  }
  
  
  
  /* Project post-collision modes back onto the distribution */
  
  for (p = 0; p < NVEL; p++) {
    DECLARE_SIMD_SCALAR(double,ftmp);
    TARGET_ILP(iv) SIMD_SC_ELMNT(ftmp,iv)=0.;
    for (m = 0; m < tc_nmodes_; m++) {
      TARGET_ILP(iv) SIMD_SC_ELMNT(ftmp,iv) += tc_mi_[p][m]*SIMD_1D_ELMNT(mode,m,iv);
    }
    TARGET_ILP(iv) t_f[ LB_ADDR(tc_nSites, NDIST, 
				      NVEL, baseIndex+iv, 
				      0, p) ] = SIMD_SC_ELMNT(ftmp,iv);
  }
  
  


  /* Now, the order parameter distribution */
  TARGET_ILP(iv)
    SIMD_SC_ELMNT(phi,iv)=t_f[ LB_ADDR(tc_nSites, NDIST, NVEL, baseIndex+iv, 1, 0) ];
  
  
  //HACK TODO vectorise this
  // SIMD_SC_ELMNT(mu) = chemical_potential(baseIndex, 0);
  
  TARGET_ILP(iv){
    SIMD_SC_ELMNT(mu,iv) = (*t_chemical_potential)(baseIndex+iv, 0,t_phi,t_delsqphi);
  
    SIMD_1D_ELMNT(jphi,X,iv) = 0.0;
    SIMD_1D_ELMNT(jphi,Y,iv) = 0.0;
    SIMD_1D_ELMNT(jphi,Z,iv) = 0.0;
  }

  for (p = 1; p < NVEL; p++) {
    TARGET_ILP(iv) SIMD_SC_ELMNT(phi,iv) += 
      t_f[ LB_ADDR(tc_nSites, NDIST, NVEL, baseIndex+iv, 1, p) ];
    for (i = 0; i < 3; i++) {
      TARGET_ILP(iv) SIMD_1D_ELMNT(jphi,i,iv) += 
	t_f[ LB_ADDR(tc_nSites, NDIST, NVEL, baseIndex+iv, 1, p) ]
	*tc_cv[p][i];
    }
  }
  
  
  /* Relax order parameters modes. See the comments above. */
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      TARGET_ILP(iv) 
	SIMD_2D_ELMNT(sphi,i,j,iv) = SIMD_SC_ELMNT(phi,iv)*SIMD_1D_ELMNT(u,i,iv)*SIMD_1D_ELMNT(u,j,iv) + SIMD_SC_ELMNT(mu,iv)*tc_d_[i][j];
      /* sphi[i][j] = phi*u[i]*u[j] + cs2*mobility*mu*d_[i][j];*/
    }
    TARGET_ILP(iv)  SIMD_1D_ELMNT(jphi,i,iv) = SIMD_1D_ELMNT(jphi,i,iv) 
      - tc_rtau2*(SIMD_1D_ELMNT(jphi,i,iv) - SIMD_SC_ELMNT(phi,iv)*SIMD_1D_ELMNT(u,i,iv));
    /* jphi[i] = phi*u[i];*/
  }
  
  /* Now update the distribution */
  
  for (p = 0; p < NVEL; p++) {
    
    int dp0 = (p == 0);

    TARGET_ILP(iv) {
      SIMD_SC_ELMNT(jdotc,iv)    = 0.0;
      SIMD_SC_ELMNT(sphidotq,iv) = 0.0;
    }
    
    for (i = 0; i < 3; i++) {
      TARGET_ILP(iv)  SIMD_SC_ELMNT(jdotc,iv) += SIMD_1D_ELMNT(jphi,i,iv)*tc_cv[p][i];
      for (j = 0; j < 3; j++) {
	TARGET_ILP(iv)  SIMD_SC_ELMNT(sphidotq,iv) += SIMD_2D_ELMNT(sphi,i,j,iv)*tc_q_[p][i][j];
      }
    }
    
    /* Project all this back to the distributions. The magic
     * here is to move phi into the non-propagating distribution. */
    
    TARGET_ILP(iv) 
      t_f[ LB_ADDR(tc_nSites, NDIST, NVEL, baseIndex+iv, 1, p) ] 
      = tc_wv[p]*(SIMD_SC_ELMNT(jdotc,iv)*tc_rcs2 
		  + SIMD_SC_ELMNT(sphidotq,iv)*tc_r2rcs4) 
      + SIMD_SC_ELMNT(phi,iv)*dp0;
    
  }
  
  return;
  
}


// full lattice operation
TARGET_ENTRY void lb_collision_binary_lattice( double* __restrict__ t_f, 
			      const double* __restrict__ t_force, 
			      double* __restrict__ t_velocity,
			      double* __restrict__ t_phi,
			      double* __restrict__ t_gradphi,
			      double* __restrict__ t_delsqphi,
			      pth_fntype* t_chemical_stress,
			      mu_fntype* t_chemical_potential, 
					       noise_t * noise, int noise_on, int nSites){

 
  int baseIndex=0;

  //partition binary collision kernel across the lattice on the target
  TARGET_TLP(baseIndex,nSites){
	
    lb_collision_binary_site( t_f, t_force, t_velocity,t_phi,t_gradphi,t_delsqphi,t_chemical_stress,t_chemical_potential,noise,noise_on,baseIndex);
        
  }
  
  
  return;
}

int lb_collision_binary(lb_t * lb, hydro_t * hydro, map_t * map, noise_t * noise) {

  int       nlocal[3];
  int noise_on = 0;                  /* Fluctuations switch */

  //  double    mode[NVEL];              /* Modes; hydrodynamic + ghost */

  double    rtau2;
  double    mobility;
  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  double    force_global[3];



  assert (NDIM == 3);
  assert(lb);
  assert(hydro);
  assert(map);

  coords_nlocal(nlocal);
  physics_fbody(force_global);


  noise_present(noise, NOISE_RHO, &noise_on);


  /* The lattice mobility gives tau = (M rho_0 / Delta t) + 1 / 2,
   * or with rho_0 = 1 etc: (1 / tau) = 2 / (2M + 1) */

  physics_mobility(&mobility);
  rtau2 = 2.0 / (1.0 + 2.0*mobility);

  int Nall[3];
  int nhalo=coords_nhalo();
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];

  int nFields=NVEL*NDIST;


 //start constant setup

  __copyConstantToTarget__(&tc_nmodes_,&nmodes_, sizeof(int));
  __copyConstantToTarget__(&tc_nmodes_, &nmodes_, sizeof(int));
  __copyConstantToTarget__(&tc_rtau_shear, &rtau_shear, sizeof(double));
  __copyConstantToTarget__(&tc_rtau_bulk, &rtau_bulk, sizeof(double));
  __copyConstantToTarget__(&tc_r3_, &r3_, sizeof(double));
  __copyConstantToTarget__(&tc_r2rcs4, &r2rcs4, sizeof(double));
  __copyConstantToTarget__(tc_rtau_, rtau_, NVEL*sizeof(double));
  __copyConstantToTarget__(tc_wv, wv, NVEL*sizeof(double));
  __copyConstantToTarget__(tc_ma_, ma_, NVEL*NVEL*sizeof(double));
  __copyConstantToTarget__(tc_mi_, mi_, NVEL*NVEL*sizeof(double));
  __copyConstantToTarget__(tc_cv, cv, NVEL*3*sizeof(int));
  __copyConstantToTarget__(&tc_rtau2, &rtau2, sizeof(double));
  __copyConstantToTarget__(&tc_rcs2, &rcs2, sizeof(double));
  __copyConstantToTarget__(&tc_nSites,&nSites, sizeof(int));
  __copyConstantToTarget__(tc_force_global,force_global, 3*sizeof(double));
  __copyConstantToTarget__(tc_d_, d_, 3*3*sizeof(double));
  __copyConstantToTarget__(tc_q_, q_, NVEL*3*3*sizeof(double));

  checkTargetError("constants");
  //end constant setup


  //start field management


  if (!symmetric_in_use()){
    printf("Error: binary collision is only compatible with symmetric free energy\n");
    exit(1);
  }


  //TO DO tidy this up - access directly.
  symmetric_t_phi(&t_phi);
  symmetric_t_gradphi(&t_gradphi);
  symmetric_t_delsqphi(&t_delsqphi);


  //copyToTargetMaskedAoS(lb->t_f,lb->f,nSites,nFields,siteMask); 
  copyToTarget(lb->t_f,lb->f,nSites*nFields*sizeof(double)); 


  //for GPU version, we use the data already existing on the target 
  //for C version, we put data on the target (for now).
  //ultimitely GPU and C versions will follow the same pattern
  #ifndef CUDA

  double *ptr;

  symmetric_phi(&ptr);
  //copyToTargetMaskedAoS(t_phi,ptr,nSites,1,siteMask); 
  copyToTarget(t_phi,ptr,nSites*sizeof(double)); 

  symmetric_delsqphi(&ptr);
  //copyToTargetMaskedAoS(t_delsqphi,ptr,nSites,1,siteMask); 
  copyToTarget(t_delsqphi,ptr,nSites*sizeof(double)); 

  symmetric_gradphi(&ptr);
  //copyToTargetMaskedAoS(t_gradphi,ptr,nSites,3,siteMask); 
  copyToTarget(t_gradphi,ptr,nSites*3*sizeof(double)); 
  #endif


  //copyToTargetMaskedAoS(hydro->t_f,hydro->f,nSites,3,siteMask); 

  copyToTarget(hydro->t_f,hydro->f,nSites*3*sizeof(double)); 

  //end field management

  //start function pointer management
  mu_fntype* t_chemical_potential; 
  targetMalloc((void**) &t_chemical_potential, sizeof(mu_fntype));
  get_chemical_potential_target(t_chemical_potential);

  pth_fntype* t_chemical_stress; 
  targetMalloc((void**) &t_chemical_stress, sizeof(pth_fntype));
  get_chemical_stress_target(t_chemical_stress);
  //end function pointer management
 


  if (noise_on) {
    
#ifdef CUDA 
    
    printf("Error: noise_on is not yet supported for CUDA\n");
    exit(1);
    
#endif      

  }

	
  lb_collision_binary_lattice TARGET_LAUNCH(nSites) ( lb->t_f, hydro->t_f, hydro->t_u,t_phi,t_gradphi,t_delsqphi,t_chemical_stress,t_chemical_potential,noise,noise_on,nSites);


#ifdef CUDA        
    copyFromTargetHaloEdge(lb->f,lb->t_f,Nall,nFields,nhalo,TARGET_EDGE); 
  //copyFromTarget(hydro->u,hydro->t_u,nSites*3*sizeof(double)); 
#else
  copyFromTarget(lb->f,lb->t_f,nSites*nFields*sizeof(double)); 
  copyFromTarget(hydro->u,hydro->t_u,nSites*3*sizeof(double)); 
#endif

  return 0;
}

/*****************************************************************************
 *
 *  fluctuations_off
 *
 *  Return zero fluctuations for stress (shat) and ghost (ghat) modes.
 *
 *****************************************************************************/

HOST TARGET static int fluctuations_off(double shat[3][3], double ghat[NVEL]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      shat[ia][ib] = 0.0;
    }
  }

  for (ia = NHYDRO; ia < NVEL; ia++) {
    ghat[ia] = 0.0;
  }

  return 0;
}

/*****************************************************************************
 *
 *  collision_stats_kt
 *
 *  Reports the equipartition of momentum, and the actual temperature
 *  cf. the expected (input) temperature.
 *
 *****************************************************************************/

int lb_collision_stats_kt(lb_t * lb, noise_t * noise, map_t * map) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n;
  int status;

  double glocal[4];
  double gtotal[4];
  double rrho;
  double gsite[3];
  double kt;

  assert(lb);
  assert(map);
  assert(noise);

  noise_present(noise, NOISE_RHO, &status);
  if (status == 0) return 0;

  coords_nlocal(nlocal);

  glocal[X] = 0.0;
  glocal[Y] = 0.0;
  glocal[Z] = 0.0;
  glocal[3] = 0.0; /* volume of fluid */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	lb_0th_moment(lb, index, LB_RHO, &rrho);
	rrho = 1.0/rrho;
	lb_1st_moment(lb, index, LB_RHO, gsite);

	for (n = 0; n < 3; n++) {
	  glocal[n] += gsite[n]*gsite[n]*rrho;
	}

	glocal[3] += 1.0;

	/* Next cell */
      }
    }
  }

  /* Divide by the actual fluid volume. The reduction is to rank 0 in
   * pe_comm() for output. */

  MPI_Reduce(glocal, gtotal, 4, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

  for (n = 0; n < 3; n++) {
    gtotal[n] /= gtotal[3];
  }

  info("\n");
  info("Isothermal fluctuations\n");
  info("[eqipart.] %14.7e %14.7e %14.7e\n", gtotal[X], gtotal[Y], gtotal[Z]);

  physics_kt(&kt);
  kt *= NDIM;
  info("[measd/kT] %14.7e %14.7e\n", gtotal[X] + gtotal[Y] + gtotal[Z], kt);

  return 0;
}

/*****************************************************************************
 *
 *  collision_ghost_modes_on
 *
 *****************************************************************************/

void collision_ghost_modes_on(void) {

  nmodes_ = NVEL;

  return;
}

/*****************************************************************************
 *
 *  collision_ghost_modes_off
 *
 *****************************************************************************/

void collision_ghost_modes_off(void) {

  nmodes_ = NHYDRO;

  return;
}

/*****************************************************************************
 *
 *  collision_relaxation_set
 *
 *****************************************************************************/

void collision_relaxation_set(const int nrelax) {

  assert(nrelax == RELAXATION_M10 ||
         nrelax == RELAXATION_BGK ||
         nrelax == RELAXATION_TRT);

  nrelax_ = nrelax;

  return;
}

/*****************************************************************************
 *
 *  lb_collision_relaxation_times_set
 *
 *  We have:
 *     Shear viscosity eta = -rho c_s^2 dt ( 1 / lambda + 1 / 2 )
 *
 *  where lambda is the eigenvalue so that the mode relaxation is
 *
 *     m_k^* = m_k + lambda (m_k - m_k^eq)
 *
 *  Bulk viscosity = -rho0 c_s^2 dt (1/3) ( 2 / lambda + 1 )
 *
 *  Note there is an extra normalisation in the lattice fluctuations
 *  which would otherwise give effective kT = cs2
 *
 *****************************************************************************/


int lb_collision_relaxation_times_set(noise_t * noise) {

  int p;
  int noise_on = 0;
  double rho0;
  double kt;
  double eta_shear;
  double eta_bulk;
  double tau, rtau;
  double tau_s;
  double tau_b;
  double tau_g;

  assert(noise);
  noise_present(noise, NOISE_RHO, &noise_on);
  physics_rho0(&rho0);

  /* Initialise the relaxation times */
 
  physics_eta_shear(&eta_shear);
  physics_eta_bulk(&eta_bulk);

  rtau_shear = 1.0/(0.5 + eta_shear / (rho0*cs2));
  rtau_bulk  = 1.0/(0.5 + eta_bulk / (rho0*cs2));

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

    tau  = eta_shear / (rho0*cs2);
    rtau = 0.5 + 2.0*tau/(tau + 3.0/8.0);
    if (rtau > 2.0) rtau = 2.0;

    if (NVEL == 15) {
      rtau_[10] = rtau_shear;
      rtau_[11] = rtau;
      rtau_[12] = rtau;
      rtau_[13] = rtau;
      rtau_[14] = rtau_shear;
    }

    if (NVEL == 19) {
      rtau_[10] = rtau_shear;
      rtau_[14] = rtau_shear;
      rtau_[18] = rtau_shear;

      rtau_[11] = rtau;
      rtau_[12] = rtau;
      rtau_[13] = rtau;
      rtau_[15] = rtau;
      rtau_[16] = rtau;
      rtau_[17] = rtau;
    }
  }

  if (noise_on) {

    tau_s = 1.0/rtau_shear;
    tau_b = 1.0/rtau_bulk;

    /* Initialise the stress variances */

    physics_kt(&kt);
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

  return 0;
}

/*****************************************************************************
 *
 *  collision_relaxation_times
 *
 *  Return NVEL (inverse) relaxation times. This is really just for
 *  information, so I've put the bulk viscosity of the diagonal of
 *  the stress elements and the shear on the off-diagonal.
 *
 *****************************************************************************/

void collision_relaxation_times(double * tau) {

  int ia, ib;
  int mode;

  assert(nrelax_ == RELAXATION_M10);

  /* Density and momentum (modes 0, 1, .. NDIM) */

  for (ia = 0; ia <= NDIM; ia++) {
    tau[ia] = 0.0;
  }

  /* Stress */

  mode = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      if (ia == ib) tau[1 + NDIM + mode++] = rtau_shear;
      if (ia != ib) tau[1 + NDIM + mode++] = rtau_bulk;
    }
  }

  for (ia = 1; ia < NDIM; ia++) {
    for (ib = 0; ib < ia; ib++) {
      if (ia == ib) tau[1 + NDIM + mode++] = rtau_shear;
      if (ia != ib) tau[1 + NDIM + mode++] = rtau_bulk;
    }
  }

  /* Ghosts */

  for (ia = NHYDRO; ia < NVEL; ia++) {
    tau[ia] = rtau_[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  collision_fluctuations
 *
 *  Compute that fluctuating contributions to the distribution at
 *  the current index.
 *
 *  There are NDIM*(NDIM+1)/2 independent stress modes, and
 *  NVEL - NHYDRO ghost modes.
 *
 *  Note: the trace needs to be corrected if this is really 2d,
 *  hence the assertion.
 *
 *****************************************************************************/

static int collision_fluctuations(noise_t * noise, int index,
				  double shat[3][3], double ghat[NVEL]) {
  int ia;
  double tr;
  double random[NNOISE_MAX];

  assert(noise);
  assert(NNOISE_MAX >= NDIM*(NDIM+1)/2);
  assert(NNOISE_MAX >= (NVEL - NHYDRO));
  assert(NDIM == 3);

  /* Set symetric random stress matrix (elements with unit variance);
   * in practice always 3d (= 6 elements) required */

  noise_reap_n(noise, index, 6, random);

  shat[X][X] = random[0];
  shat[X][Y] = random[1];
  shat[X][Z] = random[2];

  shat[Y][X] = shat[X][Y];
  shat[Y][Y] = random[3];
  shat[Y][Z] = random[4];

  shat[Z][X] = shat[X][Z];
  shat[Z][Y] = shat[Y][Z];
  shat[Z][Z] = random[5];

  /* Compute the trace and the traceless part */

  tr = r3_*(shat[X][X] + shat[Y][Y] + shat[Z][Z]);
  shat[X][X] -= tr;
  shat[Y][Y] -= tr;
  shat[Z][Z] -= tr;

  /* Set variance of the traceless part */

  shat[X][X] *= sqrt(2.0)*var_shear;
  shat[X][Y] *= var_shear;
  shat[X][Z] *= var_shear;

  shat[Y][X] *= var_shear;
  shat[Y][Y] *= sqrt(2.0)*var_shear;
  shat[Y][Z] *= var_shear;

  shat[Z][X] *= var_shear;
  shat[Z][Y] *= var_shear;
  shat[Z][Z] *= sqrt(2.0)*var_shear;

  /* Set variance of trace and recombine... */

  tr *= (var_bulk);

  shat[X][X] += tr;
  shat[Y][Y] += tr;
  shat[Z][Z] += tr;

  /* Ghost modes */

  if (nmodes_ == NVEL) {
    noise_reap_n(noise, index, NVEL-NHYDRO, random);
    for (ia = NHYDRO; ia < nmodes_; ia++) {
      ghat[ia] = noise_var[ia]*random[ia - NHYDRO];
    }
  }

  return 0;
}
