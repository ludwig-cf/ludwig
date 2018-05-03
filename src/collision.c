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
 *  (c) 2011-2017 The University of Edinburgh
 *
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
#include "coords_s.h"
#include "physics.h"
#include "model.h"
#include "lb_model_s.h"
#include "hydro_s.h"
#include "free_energy.h"
#include "control.h"
#include "collision.h"
#include "field_s.h"
#include "map_s.h"
#include "kernel.h"
#include "timer.h"

#include "symmetric.h"

/* TODO tidy forward declarations */

__global__
void lb_collision_mrt1(kernel_ctxt_t * ktx, lb_t * lb, hydro_t * hydro,
		       map_t * map, noise_t * noise);
__global__
void lb_collision_mrt2(kernel_ctxt_t * ktx, lb_t * lb, hydro_t * hydro,
		       fe_symm_t * fe, noise_t * noise);

int lb_collision_mrt(lb_t * lb, hydro_t * hydro, map_t * map, noise_t * noise);
int lb_collision_binary(lb_t * lb, hydro_t * hydro, noise_t * noise,
			fe_symm_t * fe);

static __host__ __device__
void lb_collision_fluctuations(lb_t * lb, noise_t * noise, int index,
			       double shat[3][3], double ghat[NVEL]);
int lb_collision_noise_var_set(lb_t * lb, noise_t * noise);
static __host__ int lb_collision_parameters_commit(lb_t * lb);

static __device__
void lb_collision_mrt1_site(lb_t * lb, hydro_t * hydro, map_t * map,
			    noise_t * noise, const int index0);
static __device__
void lb_collision_mrt2_site(lb_t * lb, hydro_t * hydro, fe_symm_t * fe,
			    noise_t * noise, const int index0);

__device__ void d3q19_f2mode_chunk(double* mode, const double* __restrict__ fchunk);
__device__ void d3q19_mode2f_chunk(double* mode, double* fchunk);

__device__ void d3q19_mode2f_phi(double jdotc[NSIMDVL],
				 double sphidotq[NSIMDVL],
				 double sphi[3][3][NSIMDVL],
				 double phi[NSIMDVL],
				 double jphi[3][NSIMDVL],
				 double * f, int baseIndex);

/* Additional file scope collide time constants */

typedef struct collide_param_s collide_param_t;

struct collide_param_s {
  double force_global[3];
  double mobility;
  double rtau2;
};

static __constant__ lb_collide_param_t _lbp;
static __constant__ collide_param_t _cp;

/*****************************************************************************
 *
 *  lb_collide
 *
 *  Driver routine for the collision stage.
 *
 *  We allow hydro to be NULL, in which case there is no hydrodynamics!
 * 
 *****************************************************************************/

__host__
int lb_collide(lb_t * lb, hydro_t * hydro, map_t * map, noise_t * noise,
	       fe_t * fe) {

  int ndist;

  if (hydro == NULL) return 0;

  assert(lb);
  assert(map);

  lb_ndist(lb, &ndist);
  lb_collision_relaxation_times_set(lb);
  lb_collision_noise_var_set(lb, noise);
  lb_collide_param_commit(lb);

  if (ndist == 1) lb_collision_mrt(lb, hydro, map, noise);
  if (ndist == 2) lb_collision_binary(lb, hydro, noise, (fe_symm_t *) fe);

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_mrt
 *
 *  Single fluid collision driver (multiple relaxation time).
 *
 *****************************************************************************/

__host__ int lb_collision_mrt(lb_t * lb, hydro_t * hydro, map_t * map,
			      noise_t * noise) {
  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(lb);
  assert(hydro);
  assert(map);

  cs_nlocal(lb->cs, nlocal);

  /* Local extent */
  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(lb->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  lb_collision_parameters_commit(lb);

  TIMER_start(TIMER_COLLIDE_KERNEL);

  tdpLaunchKernel(lb_collision_mrt1, nblk, ntpb, 0, 0, ctxt->target,
		  lb->target, hydro->target, map->target, noise->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  TIMER_stop(TIMER_COLLIDE_KERNEL);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_mrt1
 *
 *  Kernel driver for thread-decomposed collision routine; this generates
 *  a loop over all lattice sites.
 *
 *****************************************************************************/

__global__
void lb_collision_mrt1(kernel_ctxt_t * ktx, lb_t * lb, hydro_t * hydro,
		       map_t * map, noise_t * noise) {
  int kindex;
  int kiter;

  kiter = kernel_vector_iterations(ktx);

  targetdp_simt_for(kindex, kiter, NSIMDVL) {
    int index0;
    index0 = kernel_baseindex(ktx, kindex);
    lb_collision_mrt1_site(lb, hydro, map, noise, index0);
  }

  return;
}

/*****************************************************************************
 *
 *  lb_collision_mrt1_site
 *
 *  Collision with (potentially) different relaxation times for each
 *  different mode.
 *
 *  This code is per lattice site. To be called from
 *  lb_collision_mrt1().
 *
 *  The matrices ma and mi project the distributions onto the
 *  modes, and vice-versa, respectively, for the current LB model.
 *
 *  The collision conserves density, and momentum (to within any
 *  body force present). The stress modes, and ghost modes, are
 *  relaxed toward their equilibrium values.
 *
 *****************************************************************************/

static __device__
void lb_collision_mrt1_site(lb_t * lb, hydro_t * hydro, map_t * map,
			    noise_t * noise, const int index0) {
  
  int p, m;                               /* velocity index */
  int ia, ib;                             /* indices ("alphabeta") */
  int iv=0;                               /* SIMD loop counter */
  double mode[NVEL*NSIMDVL];              /* Modes; hydrodynamic + ghost */
  double rho[NSIMDVL], rrho[NSIMDVL];     /* Density, reciprocal density */
  double u[3][NSIMDVL];                   /* Velocity */
  double s[3][3][NSIMDVL];                /* Stress */
  double seq[3][3][NSIMDVL];              /* Equilibrium stress */
  double shat[3][3][NSIMDVL];             /* random stress */
  double ghat[NVEL][NSIMDVL];             /* noise for ghosts */

  double force[3][NSIMDVL];               /* External force */
  double tr_s[NSIMDVL], tr_seq[NSIMDVL];  /* Vectors for stress trace */
  double fchunk[NVEL*NSIMDVL];            /* 1-d SIMD distribution vector */

  char fullchunk=1;
  char includeSite[NSIMDVL];

  const double rdim = (1.0/NDIM);         /* 1 / dimension */
  KRONECKER_DELTA_CHAR(d);                /* delta_ab */

  assert(lb);
  assert(lb->param);
  assert(hydro);

  /* Determine whether this chunk of lattice sites are all active
   * and if not, which should be included */

  __targetILP__(iv) includeSite[iv] = 1;

  __targetILP__(iv) {
    if (map->status[index0+iv] != MAP_FLUID) {
      includeSite[iv] = 0;
      fullchunk = 0;
    }
  }

  /* Default to fluctuations off; shat, ghat are zero */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) shat[ia][ib][iv] = 0.0;
    }
  }

  for (ia = 0; ia < NVEL; ia++) {
    __targetILP__(iv) ghat[ia][iv] = 0.0;
  }

  /* Load SIMD vectors for distribution and force */

  for (p = 0; p < NVEL; p++) {
    __targetILP__(iv) fchunk[p*NSIMDVL+iv] = 
      lb->f[ LB_ADDR(_lbp.nsite, 1, NVEL, index0 + iv, LB_RHO, p) ];
  }

  for (ia = 0; ia < 3; ia++) {
    __targetILP__(iv) {
      force[ia][iv] = _cp.force_global[ia] 
	+ hydro->f[addr_rank1(hydro->nsite, NHDIM, index0+iv, ia)];
    }
  }
  
  /* Compute all the modes */

#ifdef _D3Q19_
    d3q19_f2mode_chunk(mode, fchunk);
#else
    for (m = 0; m < NVEL; m++) {
      __targetILP__(iv) mode[m*NSIMDVL+iv] = 0.0;
      for (p = 0; p < NVEL; p++) {
	__targetILP__(iv) {
	  mode[m*NSIMDVL+iv] += fchunk[p*NSIMDVL+iv]*_lbp.ma[m][p];
	}
      }
    }
#endif

  /* For convenience, write out the physical modes, that is,
   * rho, NDIM components of velocity, independent components
   * of stress (upper triangle), and lower triangle. */

  __targetILP__(iv) rho[iv] = mode[0*NSIMDVL+iv];
  for (ia = 0; ia < NDIM; ia++) {
    __targetILP__(iv) u[ia][iv] = mode[(1 + ia)*NSIMDVL+iv];
  }

  m = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      __targetILP__(iv) s[ia][ib][iv] = mode[(1 + NDIM + m)*NSIMDVL+iv];
      m++;
    }
  }
    
  for (ia = 1; ia < NDIM; ia++) {
    for (ib = 0; ib < ia; ib++) {
      __targetILP__(iv) s[ia][ib][iv] = s[ib][ia][iv];
    }
  }

  /* Compute the local velocity, taking account of any body force */
    
  __targetILP__(iv) rrho[iv] = 1.0/rho[iv];

  for (ia = 0; ia < NDIM; ia++) {      
    __targetILP__(iv) {
      u[ia][iv] = rrho[iv]*(u[ia][iv] + 0.5*force[ia][iv]);  
    }
  }
   
  /* Relax stress with different shear and bulk viscosity */

  __targetILP__(iv) {
    tr_s[iv]   = 0.0;
    tr_seq[iv] = 0.0;
  }

  for (ia = 0; ia < NDIM; ia++) {
    /* Set equilibrium stress */
    for (ib = 0; ib < NDIM; ib++) {
      __targetILP__(iv) {
	seq[ia][ib][iv] = rho[iv]*u[ia][iv]*u[ib][iv];
      }
    }
    /* Compute trace */
    __targetILP__(iv){
      tr_s[iv]   += s[ia][ia][iv];
      tr_seq[iv] += seq[ia][ia][iv];
    }
  }
    
  /* Form traceless parts */
  for (ia = 0; ia < NDIM; ia++) {
    __targetILP__(iv){
      s[ia][ia][iv]   -= rdim*tr_s[iv];
      seq[ia][ia][iv] -= rdim*tr_seq[iv];
    }
  }
    
  /* Relax each mode */
  __targetILP__(iv) {
    tr_s[iv] = tr_s[iv] - _lbp.rtau[LB_TAU_BULK]*(tr_s[iv] - tr_seq[iv]);
  }

  for (ia = 0; ia < NDIM; ia++) {
    for (ib = 0; ib < NDIM; ib++) {
      __targetILP__(iv) {
	s[ia][ib][iv] -= _lbp.rtau[LB_TAU_SHEAR]*(s[ia][ib][iv] - seq[ia][ib][iv]);
	s[ia][ib][iv] += d[ia][ib]*rdim*tr_s[iv];
	  
	/* Correction from body force (assumes equal relaxation times) */
	      
	s[ia][ib][iv] += (2.0 - _lbp.rtau[LB_TAU_SHEAR])
	  *(u[ia][iv]*force[ib][iv] + force[ia][iv]*u[ib][iv]);
      }
    }
  }

  if (noise->on[NOISE_RHO]) {
	
    double shat1[3][3];
    double ghat1[NVEL];

    /* This does not vectorise at the moment. Needs revisiting.
     * Note that there is a mask here to prevent random number
     * generation at solid sites, which is to maintain results
     * in regression tests. Not strictly necessary. */

    for (iv = 0; iv < NSIMDVL; iv++) {

      if (includeSite[iv]) {

	lb_collision_fluctuations(lb, noise, index0 + iv, shat1, ghat1);

	for (ia = 0; ia < NDIM; ia++) {
	  for (ib = 0; ib < NDIM; ib++) {
	    shat[ia][ib][iv] = shat1[ia][ib];
	  }
	}

	for (ia = 0; ia < NVEL; ia++) {
	  ghat[ia][iv] = ghat1[ia];
	}
      }
    }
  }

  /* Now reset the hydrodynamic modes to post-collision values:
   * rho is unchanged, velocity unchanged if no force,
   * independent components of stress, and ghosts. */
    
  for (ia = 0; ia < NDIM; ia++) {
    __targetILP__(iv) mode[(1 + ia)*NSIMDVL+iv] += force[ia][iv];
  }

  m = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      __targetILP__(iv) {
	mode[(1 + NDIM + m)*NSIMDVL+iv] = s[ia][ib][iv] + shat[ia][ib][iv];
      }
      m++;
    }
  }

  /* Ghost modes are relaxed toward zero equilibrium. */

  for (m = NHYDRO; m < NVEL; m++) {  
    __targetILP__(iv) {
      mode[m*NSIMDVL+iv] = mode[m*NSIMDVL+iv]
	- lb->param->rtau[m]*(mode[m*NSIMDVL+iv] - 0.0) + ghat[m][iv];
    }
  }


  /* Project post-collision modes back onto the distribution */
#ifdef _D3Q19_
  d3q19_mode2f_chunk(mode, fchunk);
#else
    for (p = 0; p < NVEL; p++) {
      double ftmp[NSIMDVL];
      __targetILP__(iv) ftmp[iv] = 0.0;
      for (m = 0; m < NVEL; m++) {
	__targetILP__(iv) ftmp[iv] += _lbp.mi[p][m]*mode[m*NSIMDVL+iv];
      }
      __targetILP__(iv) fchunk[p*NSIMDVL+iv] = ftmp[iv];
    }
#endif

  /* Write SIMD chunks back to main arrays. */

  if (fullchunk) {
    /* distribution */
    for (p = 0; p < NVEL; p++) {
      __targetILP__(iv) { 
	lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0+iv, LB_RHO, p)] = fchunk[p*NSIMDVL+iv];
      }
    }
    /* velocity */
    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) {
	hydro->u[addr_rank1(hydro->nsite, NHDIM, index0+iv, ia)] = u[ia][iv];
      }
    }
  }
  else {
    __targetILP__(iv) {
      if (includeSite[iv]) {
	/* distribution */
	for (p = 0; p < NVEL; p++) {
	  lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0 + iv, LB_RHO, p)]
	    = fchunk[p*NSIMDVL+iv]; 
	}
	/* velocity */

	for (ia = 0; ia < 3; ia++) {
	  hydro->u[addr_rank1(hydro->nsite, NHDIM, index0 + iv, ia)] = u[ia][iv];
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lb_collision_binary
 *
 *  Driver for 2-distribution binary fluid collision retained as an
 *  option for symmetric free energy (only).
 *
 *****************************************************************************/

__host__ int lb_collision_binary(lb_t * lb, hydro_t * hydro, noise_t * noise,
				 fe_symm_t * fe) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert (NDIM == 3); /* NDIM = 2 warrants additional tests here. */
  assert(lb);
  assert(hydro);
  assert(noise);
  assert(fe);

  cs_nlocal(lb->cs, nlocal);

  /* Kernel extent */
  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(lb->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  lb_collision_parameters_commit(lb);

  TIMER_start(TIMER_COLLIDE_KERNEL);

  tdpLaunchKernel(lb_collision_mrt2, nblk, ntpb, 0, 0, ctxt->target,
		  lb->target, hydro->target, fe->target, noise->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  TIMER_stop(TIMER_COLLIDE_KERNEL);

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_mrt2
 *
 *  This is the kernel function. The actual work is on a per-site
 *  basis in lb_collision_mrt2_site as a convenience.
 *
 *****************************************************************************/

__global__ void lb_collision_mrt2(kernel_ctxt_t * ktx, lb_t * lb,
				  hydro_t * hydro, fe_symm_t * fe,
				  noise_t * noise) {
  int kindex;
  int kiter;

  kiter = kernel_vector_iterations(ktx);

  targetdp_simt_for(kindex, kiter, NSIMDVL) {
    int index0;
    index0 = kernel_baseindex(ktx, kindex);
    lb_collision_mrt2_site(lb, hydro, fe, noise, index0);
  }

  return;
}

/*****************************************************************************
 *
 *  lb_collision_mrt2_site
 *
 *  Binary LB collision stage.
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

#define NDIST 2 /* for binary collision */

__device__ void lb_collision_mrt2_site(lb_t * lb, hydro_t * hydro,
				      fe_symm_t * fe, noise_t * noise,
				      const int index0) {
  int ia, ib, m, p;
  double f[NVEL*NSIMDVL];
  double mode[NVEL*NSIMDVL];    /* Modes; hydrodynamic + ghost */
  double rho[NSIMDVL]; 
  double rrho[NSIMDVL];         /* Density, reciprocal density */
    
  double u[3][NSIMDVL];         /* Velocity */
  double s[3][3][NSIMDVL];      /* Stress */
  double seq[3][3][NSIMDVL];    /* equilibrium stress */
  double shat[3][3][NSIMDVL];   /* random stress */
  double ghat[NVEL][NSIMDVL];   /* noise for ghosts */
  
  double force[3][NSIMDVL];     /* External force */

  double tr_s[NSIMDVL];         /* Trace of stress */ 
  double tr_seq[NSIMDVL];       /* Equilibrium value thereof */
  double phi[NSIMDVL];          /* phi */ 
  double jphi[3][NSIMDVL];      /* phi flux */
  double jdotc[NSIMDVL];        /* Contraction jphi_a cv_ia */
  double sphidotq[NSIMDVL];     /* phi second moment */
  double sth[3][3][NSIMDVL];    /* stress */
  double sphi[3][3][NSIMDVL];   /* stress */
  double mu[NSIMDVL];           /* Chemical potential */

  const double r3 = 1.0/3.0;
  KRONECKER_DELTA_CHAR(d);

  /* index for SIMD vectors */
  int iv=0;        

  assert(lb);
  assert(hydro);

  /* switch fluctuations off */
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) shat[ia][ib][iv] = 0.0;
    }
  }

  for (m = NHYDRO; m < NVEL; m++) {
    __targetILP__(iv) ghat[m][iv] = 0.0;
  }


#ifdef _D3Q19_
  for (p = 0; p < NVEL; p++) {
    targetdp_simd_for(iv, NSIMDVL) {
      f[p*NSIMDVL+iv]
	= lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0 + iv, LB_RHO, p)];
    }
  }
  d3q19_f2mode_chunk(mode, f);
#else
  /* Compute all the modes */
  for (m = 0; m < NVEL; m++) {
    targetdp_simd_for(iv, NSIMDVL) {
      mode[m*NSIMDVL+iv] = 0.0;
    }
    for (p = 0; p < NVEL; p++) {
      targetdp_simd_for(iv, NSIMDVL) {
	mode[m*NSIMDVL+iv] += _lbp.ma[m][p]
	  *lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0 + iv, LB_RHO, p)];
      }
    }
  }
#endif

  /* For convenience, write out the physical modes. */
  
  __targetILP__(iv) rho[iv] = mode[0*VVL+iv];
  for (ia = 0; ia < 3; ia++) {
    __targetILP__(iv) u[ia][iv] = mode[(1 + ia)*VVL+iv];
  }

  m = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      __targetILP__(iv) s[ia][ib][iv] = mode[(1 + NDIM + m)*VVL+iv];
      m++;
    }
  }

  for (ia = 1; ia < NDIM; ia++) {
    for (ib = 0; ib < ia; ib++) {
      __targetILP__(iv) s[ia][ib][iv] = s[ib][ia][iv];
    }
  }

  /* Compute the local velocity, taking account of any body force */
  
  __targetILP__(iv) rrho[iv] = 1.0/rho[iv];
  
  for (ia = 0; ia < 3; ia++) {
    __targetILP__(iv) {
      force[ia][iv] = _cp.force_global[ia] 
	+ hydro->f[addr_rank1(hydro->nsite, NHDIM, index0+iv, ia)];
      u[ia][iv] = rrho[iv]*(u[ia][iv] + 0.5*force[ia][iv]);  
    }
  }

  for (ia = 0; ia < 3; ia++) {   
    __targetILP__(iv) {
      hydro->u[addr_rank1(hydro->nsite, NHDIM, index0+iv, ia)] = u[ia][iv];
    }
  }
  
  /* Compute the thermodynamic component of the stress */

  fe_symm_str_v(fe, index0, sth);

  /* Relax stress with different shear and bulk viscosity */
  
  __targetILP__(iv) {
    tr_s[iv]   = 0.0;
    tr_seq[iv] = 0.0;
  }
  
  for (ia = 0; ia < 3; ia++) {
    /* Set equilibrium stress, which includes thermodynamic part */
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) {
	seq[ia][ib][iv] = rho[iv]*u[ia][iv]*u[ib][iv] + sth[ia][ib][iv];
      }
    }
    /* Compute trace */
    __targetILP__(iv) {
      tr_s[iv]   += s[ia][ia][iv];
      tr_seq[iv] += seq[ia][ia][iv];
    }
  }
  
  /* Form traceless parts */
  for (ia = 0; ia < 3; ia++) {
    __targetILP__(iv) {
      s[ia][ia][iv]   -= r3*tr_s[iv];
      seq[ia][ia][iv] -= r3*tr_seq[iv];
    }
  }

  
  /* Relax each mode */
  __targetILP__(iv)
    tr_s[iv] = tr_s[iv] - _lbp.rtau[LB_TAU_BULK]*(tr_s[iv] - tr_seq[iv]);
  
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      __targetILP__(iv) {
	s[ia][ib][iv] -= _lbp.rtau[LB_TAU_SHEAR]*(s[ia][ib][iv] - seq[ia][ib][iv]);
	s[ia][ib][iv] += d[ia][ib]*r3*tr_s[iv];
      
	/* Correction from body force (assumes equal relaxation times) */
      
	s[ia][ib][iv] += (2.0 - _lbp.rtau[LB_TAU_SHEAR])
	               *(u[ia][iv]*force[ib][iv] + force[ia][iv]*u[ib][iv]);
      }
    }
  }

  if (noise->on[NOISE_RHO]) {

    /* Not vectorised */

    for (iv = 0; iv < NSIMDVL; iv++) {
      
      double shat1[3][3];
      double ghat1[NVEL];

      lb_collision_fluctuations(lb, noise, index0 + iv, shat1, ghat1);

      for (ia = 0; ia < 3; ia++) {
	for (ib = 0; ib < 3; ib++) {
	  shat[ia][ib][iv] = shat1[ia][ib];
	}
      }
      for (p = 0; p < NVEL; p++) {
	ghat[p][iv] = ghat1[p];
      }
    }    
  }    
  
  /* Now reset the hydrodynamic modes to post-collision values:
   * rho is unchanged, velocity unchanged if no force,
   * independent components of stress, and ghosts. */
    
  for (ia = 0; ia < NDIM; ia++) {
    __targetILP__(iv) mode[(1 + ia)*VVL+iv] += force[ia][iv];
  }

  m = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      __targetILP__(iv) {
	mode[(1 + NDIM + m)*VVL+iv] = s[ia][ib][iv] + shat[ia][ib][iv];
      }
      m++;
    }
  }

  /* Ghost modes are relaxed toward zero equilibrium. */

  for (m = NHYDRO; m < NVEL; m++) { 
    __targetILP__(iv)  {
      mode[m*VVL+iv] = mode[m*VVL+iv] 
	- lb->param->rtau[m]*(mode[m*VVL+iv] - 0.0) + ghat[m][iv];
    }
  }

  /* Project post-collision modes back onto the distribution */

#ifdef _D3Q19_  
  d3q19_mode2f_chunk(mode, f);
  for (p = 0; p < NVEL; p++) {
    targetdp_simd_for(iv, NSIMDVL) {
      lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0 + iv, LB_RHO, p)] =
	f[p*VVL+iv];
    }
  }
#else    
  for (p = 0; p < NVEL; p++) {
    __targetILP__(iv) f[p*NSIMDVL+iv] = 0.0;
    for (m = 0; m < NVEL; m++) {
      __targetILP__(iv) f[p*NSIMDVL+iv] += _lbp.mi[p][m]*mode[m*NSIMDVL+iv];
    }
    __targetILP__(iv) {
      lb->f[LB_ADDR(_lbp.nsite, NDIST, NVEL, index0+iv, LB_RHO, p)] = f[p*NSIMDVL+iv];
    }
  }
#endif

  /* Now, the order parameter distribution */
  __targetILP__(iv) {
    phi[iv] = fe->phi->data[addr_rank0(fe->phi->nsites, index0 + iv)];
  }

  __targetILP__(iv) {
    fe_symm_mu(fe, index0 + iv, mu + iv);
    jphi[X][iv] = 0.0;
    jphi[Y][iv] = 0.0;
    jphi[Z][iv] = 0.0;
  }

  for (p = 1; p < NVEL; p++) {
    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) {
	jphi[ia][iv] += _lbp.cv[p][ia]* 
	lb->f[ LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0+iv, LB_PHI, p) ];
      }
    }
  }

  /* Relax order parameter modes. See the comments above. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      __targetILP__(iv) { 
	sphi[ia][ib][iv] = phi[iv]*u[ia][iv]*u[ib][iv] + mu[iv]*d[ia][ib];
        /* sphi[ia][ib] = phi*u[ia]*u[ib] + cs2*mobility*mu*d_[ia][ib];*/
      }
    }
    __targetILP__(iv) {
      jphi[ia][iv] = jphi[ia][iv] - _cp.rtau2*(jphi[ia][iv] - phi[iv]*u[ia][iv]);
      /* jphi[ia] = phi*u[ia]; */
    }
  }
  
  /* Now update the distribution */
  
#ifdef _D3Q19_
  d3q19_mode2f_phi(jdotc,sphidotq,sphi,phi,jphi, lb->f, index0);
#else

  for (p = 0; p < NVEL; p++) {
    
    int dp0 = (p == 0);

    __targetILP__(iv) {
      jdotc[iv]    = 0.0;
      sphidotq[iv] = 0.0;
    }
    
    for (ia = 0; ia < 3; ia++) {
      __targetILP__(iv) jdotc[iv] += jphi[ia][iv]*_lbp.cv[p][ia];
      for (ib = 0; ib < 3; ib++) {
	__targetILP__(iv) sphidotq[iv] += sphi[ia][ib][iv]*_lbp.q[p][ia][ib];
      }
    }
    
    /* Project all this back to the distributions. The magic
     * here is to move phi into the non-propagating distribution. */
    __targetILP__(iv) { 
      lb->f[ LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0+iv, LB_PHI, p) ] 
      = _lbp.wv[p]*(jdotc[iv]*3.0 + sphidotq[iv]*4.5) + phi[iv]*dp0;
    }
  }
#endif

  return;
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
  physics_t * phys = NULL;
  MPI_Comm comm;

  assert(lb);
  assert(map);
  assert(noise);

  noise_present(noise, NOISE_RHO, &status);
  if (status == 0) return 0;

  physics_ref(&phys);
  physics_kt(phys, &kt);

  cs_nlocal(lb->cs, nlocal);

  glocal[X] = 0.0;
  glocal[Y] = 0.0;
  glocal[Z] = 0.0;
  glocal[3] = 0.0; /* volume of fluid */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(lb->cs, ic, jc, kc);
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
   * pe_mpi_comm() for output. */

  pe_mpi_comm(lb->pe, &comm);
  MPI_Reduce(glocal, gtotal, 4, MPI_DOUBLE, MPI_SUM, 0, comm);

  for (n = 0; n < 3; n++) {
    gtotal[n] /= gtotal[3];
  }

  pe_info(lb->pe, "\n");
  pe_info(lb->pe, "Isothermal fluctuations\n");
  pe_info(lb->pe, "[eqipart.] %14.7e %14.7e %14.7e\n", gtotal[X], gtotal[Y],
	  gtotal[Z]);

  kt *= NDIM;
  pe_info(lb->pe, "[measd/kT] %14.7e %14.7e\n",
	  gtotal[X] + gtotal[Y] + gtotal[Z], kt);

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_ghost_modes_on
 *
 *****************************************************************************/

 __host__ int lb_collision_ghost_modes_on(lb_t * lb) {

   assert(lb);
   assert(lb->param);

  lb->param->isghost = LB_GHOST_ON;

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_ghost_modes_off
 *
 *****************************************************************************/

 __host__ int lb_collision_ghost_modes_off(lb_t * lb) {

   assert(lb);
   assert(lb->param);

  lb->param->isghost = LB_GHOST_OFF;

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_relaxation_set
 *
 *****************************************************************************/

__host__ int lb_collision_relaxation_set(lb_t * lb, lb_relaxation_enum_t nrelax) {

  assert(nrelax == LB_RELAXATION_M10 ||
         nrelax == LB_RELAXATION_BGK ||
         nrelax == LB_RELAXATION_TRT);

  assert(lb);

  lb->nrelax = nrelax;

  return 0;
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
 *****************************************************************************/

__host__ int lb_collision_relaxation_times_set(lb_t * lb) {

  int p;
  double rho0;
  double eta_shear;
  double eta_bulk;
  double rtau_shear;
  double rtau_bulk;
  double tau, rtau;
  physics_t * phys = NULL;

  assert(lb);
  assert(lb->param);

  physics_ref(&phys);
  physics_rho0(phys, &rho0);

  lb->param->rho0 = rho0;

  /* Initialise the relaxation times */
 
  physics_eta_shear(phys, &eta_shear);
  physics_eta_bulk(phys, &eta_bulk);

  rtau_shear = 1.0/(0.5 + eta_shear / (rho0*cs2));
  rtau_bulk  = 1.0/(0.5 + eta_bulk / (rho0*cs2));

  if (lb->nrelax == LB_RELAXATION_M10) {
    lb->param->rtau[LB_TAU_SHEAR] = rtau_shear;
    lb->param->rtau[LB_TAU_BULK]  = rtau_bulk;
    for (p = NHYDRO; p < NVEL; p++) {
      lb->param->rtau[p] = 1.0;
    }
  }

  if (lb->nrelax == LB_RELAXATION_BGK) {
    lb->param->rtau[LB_TAU_SHEAR] = rtau_shear;
    lb->param->rtau[LB_TAU_BULK]  = rtau_shear; /* No separate bulk visocity */
    for (p = 0; p < NVEL; p++) {
      lb->param->rtau[p] = rtau_shear;
    }
  }

  if (lb->nrelax == LB_RELAXATION_TRT) {

    assert(NVEL != 9);

    lb->param->rtau[LB_TAU_SHEAR] = rtau_shear;
    lb->param->rtau[LB_TAU_BULK]  = rtau_bulk;

    tau  = eta_shear / (rho0*cs2);
    rtau = 0.5 + 2.0*tau/(tau + 3.0/8.0);
    if (rtau > 2.0) rtau = 2.0;

    if (NVEL == 15) {
      lb->param->rtau[10] = rtau_shear;
      lb->param->rtau[11] = rtau;
      lb->param->rtau[12] = rtau;
      lb->param->rtau[13] = rtau;
      lb->param->rtau[14] = rtau_shear;
    }

    if (NVEL == 19) {
      lb->param->rtau[10] = rtau_shear;
      lb->param->rtau[14] = rtau_shear;
      lb->param->rtau[18] = rtau_shear;

      lb->param->rtau[11] = rtau;
      lb->param->rtau[12] = rtau;
      lb->param->rtau[13] = rtau;
      lb->param->rtau[15] = rtau;
      lb->param->rtau[16] = rtau;
      lb->param->rtau[17] = rtau;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_noise_var_set
 *
 *  Note there is an extra normalisation in the lattice fluctuations
 *  which would otherwise give effective kT = cs2
 *
 *****************************************************************************/

__host__ int lb_collision_noise_var_set(lb_t * lb, noise_t * noise) {

  int p;
  int noise_on = 0;
  double kt;
  double tau_s;
  double tau_b;
  double tau_g;
  physics_t * phys = NULL;

  assert(lb);
  assert(noise);

  noise_present(noise, NOISE_RHO, &noise_on);

  if (noise_on) {

    physics_ref(&phys);
    physics_kt(phys, &kt);

    tau_s = 1.0/lb->param->rtau[LB_TAU_SHEAR];
    tau_b = 1.0/lb->param->rtau[LB_TAU_BULK];

    /* Initialise the stress variances */

    physics_kt(phys, &kt);
    kt = kt*rcs2; /* Without normalisation kT = cs^2 */

    lb->param->var_bulk =
      sqrt(kt)*sqrt(2.0/9.0)*sqrt((tau_b + tau_b - 1.0)/(tau_b*tau_b));
    lb->param->var_shear =
      sqrt(kt)*sqrt(1.0/9.0)*sqrt((tau_s + tau_s - 1.0)/(tau_s*tau_s));

    /* Noise variances */

    for (p = NHYDRO; p < NVEL; p++) {
      tau_g = 1.0/lb->param->rtau[p];
      lb->param->var_noise[p] =
	sqrt(kt/norm_[p])*sqrt((tau_g + tau_g - 1.0)/(tau_g*tau_g));
    }
  }

  if (lb->param->isghost == LB_GHOST_OFF) {
    /* This option is intended to check the M10 without the correct
     * noise terms. Should not be used for a real simulation. */
    /* Eliminate ghost modes and ghost mode noise */
    for (p = NHYDRO; p < NVEL; p++) {
      lb->param->rtau[p] = 1.0;
      lb->param->var_noise[p] = 0.0;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_relaxation_times
 *
 *  Return NVEL relaxation times. This is really just for information.
 *  The relaxation times are computed, so viscosities must be available.
 *
 *****************************************************************************/

__host__ int lb_collision_relaxation_times(lb_t * lb, double * tau) {

  int ia;

  assert(lb);
  assert(tau);

  /* Density and momentum (modes 0, 1, .. NDIM) */

  for (ia = 0; ia <= NDIM; ia++) {
    tau[ia] = 0.0;
  }

  lb_collision_relaxation_times_set(lb);

  for (ia = NDIM+1; ia < NVEL; ia++) {
    tau[ia] = 1.0/lb->param->rtau[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_fluctuations
 *
 *  Compute that fluctuating contributions to the distribution at
 *  the current lattice site index.
 *
 *  There are NDIM*(NDIM+1)/2 independent stress modes, and
 *  NVEL - NHYDRO ghost modes.
 *
 *****************************************************************************/

static __host__ __device__
  void lb_collision_fluctuations(lb_t * lb, noise_t * noise, int index,
				 double shat[3][3], double ghat[NVEL]) {
  int ia;
  double tr;
  double random[NNOISE_MAX];

  assert(lb);
  assert(lb->param);
  assert(noise);
  assert(NNOISE_MAX >= NDIM*(NDIM+1)/2);
  assert(NNOISE_MAX >= (NVEL - NHYDRO));
  assert(NDIM == 2 || NDIM == 3);

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

  tr = (1.0/NDIM)*(shat[X][X] + shat[Y][Y] + (NDIM - 2.0)*shat[Z][Z]);
  shat[X][X] -= tr;
  shat[Y][Y] -= tr;
  shat[Z][Z] -= tr;

  /* Set variance of the traceless part */

  shat[X][X] *= lb->param->var_shear*sqrt(2.0);
  shat[X][Y] *= lb->param->var_shear;
  shat[X][Z] *= lb->param->var_shear;

  shat[Y][X] *= lb->param->var_shear;
  shat[Y][Y] *= lb->param->var_shear*sqrt(2.0);
  shat[Y][Z] *= lb->param->var_shear;

  shat[Z][X] *= lb->param->var_shear;
  shat[Z][Y] *= lb->param->var_shear;
  shat[Z][Z] *= lb->param->var_shear*sqrt(2.0);

  /* Set variance of trace and recombine... */

  tr *= (lb->param->var_bulk);

  shat[X][X] += tr;
  shat[Y][Y] += tr;
  shat[Z][Z] += tr;

  /* Ghost modes */

  for (ia = 0; ia < NHYDRO; ia++) {
    ghat[ia] = 0.0;
  }

  if (lb->param->isghost == LB_GHOST_ON) {
    noise_reap_n(noise, index, NVEL-NHYDRO, random);

    for (ia = NHYDRO; ia < NVEL; ia++) {
      ghat[ia] = lb->param->var_noise[ia]*random[ia - NHYDRO];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lb_collision_parameters_commit
 *
 *****************************************************************************/

static __host__ int lb_collision_parameters_commit(lb_t * lb) {

  collide_param_t p;
  physics_t * phys = NULL;

  assert(lb);

  physics_ref(&phys);
  physics_fbody(phys, p.force_global);

  /* The lattice mobility gives tau = (M rho_0 / Delta t) + 1 / 2,
   * or with rho_0 = 1 etc: (1 / tau) = 2 / (2M + 1) */

  physics_mobility(phys, &p.mobility);
  p.rtau2 = 2.0 / (1.0 + 2.0*p.mobility);

  tdpMemcpyToSymbol(tdpSymbol(_lbp), lb->param, sizeof(lb_collide_param_t),
		    0, tdpMemcpyHostToDevice);
  tdpMemcpyToSymbol(tdpSymbol(_cp), &p, sizeof(collide_param_t), 0,
		    tdpMemcpyHostToDevice);
  return 0;
}



#ifdef _D3Q19_

/*  below are specialized fast unrolled version of the d3q19 19x19 matrix 
 *  multiplications. These are significantly faster because there is 
 * duplication in the 19x19 matrix. Explicitly coding the values means 
 * less data movement on the chip. */

#define w0 (12.0/36.0)
#define w1  (2.0/36.0)
#define w2  (1.0/36.0)

#define c0        0.0
#define c1        1.0
#define c2        2.0
#define r3   (1.0/3.0)
#define r6   (1.0/6.0)
#define t3   (2.0/3.0)
#define r2   (1.0/2.0)
#define r4   (1.0/4.0)
#define r8   (1.0/8.0)

#define wc ( 1.0/72.0)
#define wb ( 3.0/72.0)
#define wa ( 6.0/72.0)
#define t4 (16.0/72.0)
#define wd ( 1.0/48.0)
#define we ( 3.0/48.0)

__device__ void d3q19_f2mode_chunk(double* mode, const double* __restrict__ fchunk)
{

  int m, il=0;

   for (m = 0; m < NVEL; m++) { 
       __targetILP__(il) mode[m*VVL+il] = 0.0; 
   }


  __targetILP__(il) mode[0*VVL+il] += fchunk[0*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[1*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[2*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[3*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[4*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[5*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[7*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[8*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[9*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[10*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[11*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[12*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[13*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[14*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[15*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[16*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[17*VVL+il]*c1;
  __targetILP__(il) mode[0*VVL+il] += fchunk[18*VVL+il]*c1;

  /* m=1*/
  __targetILP__(il) mode[1*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[1*VVL+il]*c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[2*VVL+il]*c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[3*VVL+il]*c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[4*VVL+il]*c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[5*VVL+il]*c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[6*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[8*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[11*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[13*VVL+il]*c0;
  __targetILP__(il) mode[1*VVL+il] += fchunk[14*VVL+il]*-c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[15*VVL+il]*-c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[16*VVL+il]*-c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[17*VVL+il]*-c1;
  __targetILP__(il) mode[1*VVL+il] += fchunk[18*VVL+il]*-c1;

  /* m=2*/
  __targetILP__(il) mode[2*VVL+il]=0.;
  __targetILP__(il) mode[2*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[1*VVL+il]*c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[2*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[4*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[5*VVL+il]*-c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[7*VVL+il]*c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[8*VVL+il]*c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[11*VVL+il]*-c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[12*VVL+il]*-c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[13*VVL+il]*-c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[14*VVL+il]*c1;
  __targetILP__(il) mode[2*VVL+il] += fchunk[15*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[17*VVL+il]*c0;
  __targetILP__(il) mode[2*VVL+il] += fchunk[18*VVL+il]*-c1;

  /* m=3*/
  __targetILP__(il) mode[3*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[3*VVL+il] += fchunk[1*VVL+il]*c0;
  __targetILP__(il) mode[3*VVL+il] += fchunk[2*VVL+il]*c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[3*VVL+il] += fchunk[4*VVL+il]*-c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[5*VVL+il]*c0;
  __targetILP__(il) mode[3*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[3*VVL+il] += fchunk[8*VVL+il]*-c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[9*VVL+il]*c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[10*VVL+il]*-c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[11*VVL+il]*c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[3*VVL+il] += fchunk[13*VVL+il]*-c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[14*VVL+il]*c0;
  __targetILP__(il) mode[3*VVL+il] += fchunk[15*VVL+il]*c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[3*VVL+il] += fchunk[17*VVL+il]*-c1;
  __targetILP__(il) mode[3*VVL+il] += fchunk[18*VVL+il]*c0;

  /* m=4*/
  __targetILP__(il) mode[4*VVL+il] += fchunk[0*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[1*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[2*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[3*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[4*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[5*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[6*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[7*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[8*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[9*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[10*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[11*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[12*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[13*VVL+il]*-r3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[14*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[15*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[16*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[17*VVL+il]*t3;
  __targetILP__(il) mode[4*VVL+il] += fchunk[18*VVL+il]*t3;

  /* m=5*/
  __targetILP__(il) mode[5*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[1*VVL+il]*c1;
  __targetILP__(il) mode[5*VVL+il] += fchunk[2*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[4*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[5*VVL+il]*-c1;
  __targetILP__(il) mode[5*VVL+il] += fchunk[6*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[8*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[11*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[13*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[14*VVL+il]*-c1;
  __targetILP__(il) mode[5*VVL+il] += fchunk[15*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[17*VVL+il]*c0;
  __targetILP__(il) mode[5*VVL+il] += fchunk[18*VVL+il]*c1;

  /* m=6*/
  __targetILP__(il) mode[6*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[1*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[2*VVL+il]*c1;
  __targetILP__(il) mode[6*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[4*VVL+il]*-c1;
  __targetILP__(il) mode[6*VVL+il] += fchunk[5*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[6*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[8*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[11*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[13*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[14*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[15*VVL+il]*-c1;
  __targetILP__(il) mode[6*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[6*VVL+il] += fchunk[17*VVL+il]*c1;
  __targetILP__(il) mode[6*VVL+il] += fchunk[18*VVL+il]*c0;

  /* m=7*/
  __targetILP__(il) mode[7*VVL+il] += fchunk[0*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[1*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[2*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[3*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[4*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[5*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[6*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[7*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[8*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[9*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[10*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[11*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[12*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[13*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[14*VVL+il]*t3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[15*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[16*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[17*VVL+il]*-r3;
  __targetILP__(il) mode[7*VVL+il] += fchunk[18*VVL+il]*t3;

  /* m=8*/
  __targetILP__(il) mode[8*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[1*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[2*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[4*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[5*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[8*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[8*VVL+il]*-c1;
  __targetILP__(il) mode[8*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[11*VVL+il]*-c1;
  __targetILP__(il) mode[8*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[13*VVL+il]*c1;
  __targetILP__(il) mode[8*VVL+il] += fchunk[14*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[15*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[17*VVL+il]*c0;
  __targetILP__(il) mode[8*VVL+il] += fchunk[18*VVL+il]*c0;

  /* m=9*/
  __targetILP__(il) mode[9*VVL+il] += fchunk[0*VVL+il]*-r3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[1*VVL+il]*-r3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[2*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[3*VVL+il]*-r3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[4*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[5*VVL+il]*-r3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[6*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[7*VVL+il]*-r3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[8*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[9*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[10*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[11*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[12*VVL+il]*-r3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[13*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[14*VVL+il]*-r3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[15*VVL+il]*t3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[16*VVL+il]*-r3;
  __targetILP__(il) mode[9*VVL+il] += fchunk[17*VVL+il]*t3 ;
  __targetILP__(il) mode[9*VVL+il] += fchunk[18*VVL+il]*-r3;

  /* m=10*/
  __targetILP__(il) mode[10*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[10*VVL+il] += fchunk[1*VVL+il]*-c2;
  __targetILP__(il) mode[10*VVL+il] += fchunk[2*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[3*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[4*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[5*VVL+il]*-c2;
  __targetILP__(il) mode[10*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[7*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[8*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[9*VVL+il]*-c2;
  __targetILP__(il) mode[10*VVL+il] += fchunk[10*VVL+il]*-c2;
  __targetILP__(il) mode[10*VVL+il] += fchunk[11*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[12*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[13*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[14*VVL+il]*-c2;
  __targetILP__(il) mode[10*VVL+il] += fchunk[15*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[16*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[17*VVL+il]*c1;
  __targetILP__(il) mode[10*VVL+il] += fchunk[18*VVL+il]*-c2;

  /* m=11*/
  __targetILP__(il) mode[11*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[1*VVL+il]*-c2;
  __targetILP__(il) mode[11*VVL+il] += fchunk[2*VVL+il]*c1;
  __targetILP__(il) mode[11*VVL+il] += fchunk[3*VVL+il]*c1;
  __targetILP__(il) mode[11*VVL+il] += fchunk[4*VVL+il]*c1;
  __targetILP__(il) mode[11*VVL+il] += fchunk[5*VVL+il]*-c2;
  __targetILP__(il) mode[11*VVL+il] += fchunk[6*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[8*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[11*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[13*VVL+il]*c0;
  __targetILP__(il) mode[11*VVL+il] += fchunk[14*VVL+il]*c2;
  __targetILP__(il) mode[11*VVL+il] += fchunk[15*VVL+il]*-c1;
  __targetILP__(il) mode[11*VVL+il] += fchunk[16*VVL+il]*-c1;
  __targetILP__(il) mode[11*VVL+il] += fchunk[17*VVL+il]*-c1;
  __targetILP__(il) mode[11*VVL+il] += fchunk[18*VVL+il]*c2;

  /* m=12*/
  __targetILP__(il) mode[12*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[1*VVL+il]*-c2;
  __targetILP__(il) mode[12*VVL+il] += fchunk[2*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[4*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[5*VVL+il]*c2;
  __targetILP__(il) mode[12*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[12*VVL+il] += fchunk[7*VVL+il]*c1;
  __targetILP__(il) mode[12*VVL+il] += fchunk[8*VVL+il]*c1;
  __targetILP__(il) mode[12*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[11*VVL+il]*-c1;
  __targetILP__(il) mode[12*VVL+il] += fchunk[12*VVL+il]*-c1;
  __targetILP__(il) mode[12*VVL+il] += fchunk[13*VVL+il]*-c1;
  __targetILP__(il) mode[12*VVL+il] += fchunk[14*VVL+il]*-c2;
  __targetILP__(il) mode[12*VVL+il] += fchunk[15*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[17*VVL+il]*c0;
  __targetILP__(il) mode[12*VVL+il] += fchunk[18*VVL+il]*c2;

  /* m=13*/
  __targetILP__(il) mode[13*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[1*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[2*VVL+il]*c1;
  __targetILP__(il) mode[13*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[4*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[5*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[13*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[8*VVL+il]*-c1;
  __targetILP__(il) mode[13*VVL+il] += fchunk[9*VVL+il]*-c2;
  __targetILP__(il) mode[13*VVL+il] += fchunk[10*VVL+il]*c2;
  __targetILP__(il) mode[13*VVL+il] += fchunk[11*VVL+il]*c1;
  __targetILP__(il) mode[13*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[13*VVL+il]*-c1;
  __targetILP__(il) mode[13*VVL+il] += fchunk[14*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[15*VVL+il]*c1;
  __targetILP__(il) mode[13*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[13*VVL+il] += fchunk[17*VVL+il]*-c1;
  __targetILP__(il) mode[13*VVL+il] += fchunk[18*VVL+il]*c0;

  /* m=14*/
  __targetILP__(il) mode[14*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[14*VVL+il] += fchunk[1*VVL+il]*c0;
  __targetILP__(il) mode[14*VVL+il] += fchunk[2*VVL+il]*-c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[3*VVL+il]*c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[4*VVL+il]*-c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[5*VVL+il]*c0;
  __targetILP__(il) mode[14*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[7*VVL+il]*-c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[8*VVL+il]*c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[14*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[14*VVL+il] += fchunk[11*VVL+il]*c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[12*VVL+il]*-c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[13*VVL+il]*c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[14*VVL+il]*c0;
  __targetILP__(il) mode[14*VVL+il] += fchunk[15*VVL+il]*-c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[16*VVL+il]*c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[17*VVL+il]*-c1;
  __targetILP__(il) mode[14*VVL+il] += fchunk[18*VVL+il]*c0;

  /* m=15*/
  __targetILP__(il) mode[15*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[1*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[2*VVL+il]*-c1;
  __targetILP__(il) mode[15*VVL+il] += fchunk[3*VVL+il]*c1;
  __targetILP__(il) mode[15*VVL+il] += fchunk[4*VVL+il]*-c1;
  __targetILP__(il) mode[15*VVL+il] += fchunk[5*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[6*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[8*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[11*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[13*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[14*VVL+il]*c0;
  __targetILP__(il) mode[15*VVL+il] += fchunk[15*VVL+il]*c1;
  __targetILP__(il) mode[15*VVL+il] += fchunk[16*VVL+il]*-c1;
  __targetILP__(il) mode[15*VVL+il] += fchunk[17*VVL+il]*c1;
  __targetILP__(il) mode[15*VVL+il] += fchunk[18*VVL+il]*c0;

  /* m=16*/
  __targetILP__(il) mode[16*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[1*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[2*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[4*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[5*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[16*VVL+il] += fchunk[7*VVL+il]*-c1;
  __targetILP__(il) mode[16*VVL+il] += fchunk[8*VVL+il]*c1;
  __targetILP__(il) mode[16*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[11*VVL+il]*-c1;
  __targetILP__(il) mode[16*VVL+il] += fchunk[12*VVL+il]*c1;
  __targetILP__(il) mode[16*VVL+il] += fchunk[13*VVL+il]*-c1;
  __targetILP__(il) mode[16*VVL+il] += fchunk[14*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[15*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[17*VVL+il]*c0;
  __targetILP__(il) mode[16*VVL+il] += fchunk[18*VVL+il]*c0;

  /* m=17*/
  __targetILP__(il) mode[17*VVL+il] += fchunk[0*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[1*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[2*VVL+il]*-c1;
  __targetILP__(il) mode[17*VVL+il] += fchunk[3*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[4*VVL+il]*c1;
  __targetILP__(il) mode[17*VVL+il] += fchunk[5*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[17*VVL+il] += fchunk[7*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[8*VVL+il]*-c1;
  __targetILP__(il) mode[17*VVL+il] += fchunk[9*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[10*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[11*VVL+il]*c1;
  __targetILP__(il) mode[17*VVL+il] += fchunk[12*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[13*VVL+il]*-c1;
  __targetILP__(il) mode[17*VVL+il] += fchunk[14*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[15*VVL+il]*-c1;
  __targetILP__(il) mode[17*VVL+il] += fchunk[16*VVL+il]*c0;
  __targetILP__(il) mode[17*VVL+il] += fchunk[17*VVL+il]*c1;
  __targetILP__(il) mode[17*VVL+il] += fchunk[18*VVL+il]*c0;

  /* m=18*/
  __targetILP__(il) mode[18*VVL+il] += fchunk[0*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[1*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[2*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[3*VVL+il]*-c2;
  __targetILP__(il) mode[18*VVL+il] += fchunk[4*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[5*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[6*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[7*VVL+il]*-c2;
  __targetILP__(il) mode[18*VVL+il] += fchunk[8*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[9*VVL+il]*-c2;
  __targetILP__(il) mode[18*VVL+il] += fchunk[10*VVL+il]*-c2;
  __targetILP__(il) mode[18*VVL+il] += fchunk[11*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[12*VVL+il]*-c2;
  __targetILP__(il) mode[18*VVL+il] += fchunk[13*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[14*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[15*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[16*VVL+il]*-c2;
  __targetILP__(il) mode[18*VVL+il] += fchunk[17*VVL+il]*c1;
  __targetILP__(il) mode[18*VVL+il] += fchunk[18*VVL+il]*c1;
 
}

__device__ void d3q19_mode2f_chunk(double* mode, double* fchunk) {

  double ftmp[NSIMDVL];

  int il=0;

  
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w0*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -r2*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -r2*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -r2*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[18*VVL+il];
  __targetILP__(il)   fchunk[0*VVL+il] = ftmp[il];
  
  /* p=1*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += r4*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[1*VVL+il] = ftmp[il];
  
  /* p=2*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += r4*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wd*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += -we*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[2*VVL+il] =  ftmp[il];
  
  /* p=3*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w1*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += r4*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += -w1*mode[18*VVL+il];
  __targetILP__(il) fchunk[3*VVL+il] =  ftmp[il];
  
  /* p=4*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += -r4*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wd*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += -we*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[4*VVL+il] = ftmp[il];
  
  /* p=5 */
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += -r4*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[5*VVL+il] = ftmp[il];
  
  /* p=6 */
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += r4*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wd*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += we*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[6*VVL+il] =  ftmp[il];
  
  /* p=7 */
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w1*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += -r4*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += -w1*mode[18*VVL+il];
  __targetILP__(il) fchunk[7*VVL+il] = ftmp[il];
  
  /* p=8 */
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += -r4*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wd*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += we*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[8*VVL+il] =  ftmp[il];
  
  /* p=9*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w1*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += -r6*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += -w1*mode[18*VVL+il];
  __targetILP__(il) fchunk[9*VVL+il] = ftmp[il];
  
  /* p=10*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w1*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += -r6*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += -w1*mode[18*VVL+il];
  __targetILP__(il) fchunk[10*VVL+il] = ftmp[il];
  
  /* p=11 */
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += -r4*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wd*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += we*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[11*VVL+il] =  ftmp[il];
  
  /* p=12*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w1*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += -r6*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += r4*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += -w1*mode[18*VVL+il];
  __targetILP__(il) fchunk[12*VVL+il] = ftmp[il];
  
  /* p=13 */
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += r4*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wd*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += we*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[13*VVL+il] =  ftmp[il];
  
  /* p=14 */
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += -r4*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[14*VVL+il] =  ftmp[il];
  
  /* p=15*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += -r4*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wd*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += -we*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += -r8*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[15*VVL+il] =  ftmp[il];
  
  /* p=16*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w1*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += -r6*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += r6*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wb*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += -r4*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += -w1*mode[18*VVL+il];
  __targetILP__(il) fchunk[16*VVL+il] =ftmp[il];
  
  /* p=17*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += r4*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += wd*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += -we*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += r8*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[17*VVL+il] = ftmp[il];
  
  /* p=18*/
  __targetILP__(il) ftmp[il]=0.;
  __targetILP__(il) ftmp[il] += w2*mode[0*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[1*VVL+il];
  __targetILP__(il) ftmp[il] += -wa*mode[2*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[3*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[4*VVL+il];
  __targetILP__(il) ftmp[il] += r4*mode[5*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[6*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[7*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[8*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[9*VVL+il];
  __targetILP__(il) ftmp[il] += -wb*mode[10*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[11*VVL+il];
  __targetILP__(il) ftmp[il] += wa*mode[12*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[13*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[14*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[15*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[16*VVL+il];
  __targetILP__(il) ftmp[il] += c0*mode[17*VVL+il];
  __targetILP__(il) ftmp[il] += wc*mode[18*VVL+il];
  __targetILP__(il) fchunk[18*VVL+il] = ftmp[il];


}


__device__ void d3q19_mode2f_phi(double jdotc[NSIMDVL],
				 double sphidotq[NSIMDVL],
				 double sphi[3][3][NSIMDVL],
				 double phi[NSIMDVL],
				 double jphi[3][NSIMDVL],
				 double * f, int baseIndex){

  int iv=0;
  const double rcs2 = 3.0;
  const double r2rcs4 = (9.0/2.0);

  __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

  __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 0) ] 
        = w0*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4) + phi[iv];


  __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[X][iv];
  __targetILP__(iv)  jdotc[iv] += jphi[Y][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[0][1][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][0][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 1) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[X][iv];
  __targetILP__(iv)  jdotc[iv] += jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[0][2][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][0][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 2) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[X][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 3) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[X][iv];
  __targetILP__(iv)  jdotc[iv] -= jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[0][2][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][0][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 4) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[X][iv];
  __targetILP__(iv)  jdotc[iv] -= jphi[Y][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[0][1][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][0][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 5) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[Y][iv];
  __targetILP__(iv)  jdotc[iv] += jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][2][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][1][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 6) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[Y][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 7) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[Y][iv];
  __targetILP__(iv)  jdotc[iv] -= jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][2][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][1][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 8) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] += jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 9) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 10) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[Y][iv];
  __targetILP__(iv)  jdotc[iv] += jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][2][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][1][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 11) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[Y][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 12) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[Y][iv];
  __targetILP__(iv)  jdotc[iv] -= jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][2][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][1][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 13) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[X][iv];
  __targetILP__(iv)  jdotc[iv] += jphi[Y][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[0][1][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][0][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 14) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[X][iv];
  __targetILP__(iv)  jdotc[iv] += jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[0][2][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][0][iv]*-1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 15) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[X][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 16) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[X][iv];
  __targetILP__(iv)  jdotc[iv] -= jphi[Z][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[0][2][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][0][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 17) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 __targetILP__(iv) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  __targetILP__(iv)  jdotc[iv] -= jphi[X][iv];
  __targetILP__(iv)  jdotc[iv] -= jphi[Y][iv];
  __targetILP__(iv)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[0][1][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][0][iv]*1.0000000000000000e+00;
  __targetILP__(iv)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  __targetILP__(iv)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 __targetILP__(iv) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 18) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);

  return;
}
#endif
