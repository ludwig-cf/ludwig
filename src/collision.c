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
 *  (c) 2011-2018 The University of Edinburgh
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

__global__
void lb_collision_mrt1(kernel_ctxt_t * ktx, lb_t * lb, hydro_t * hydro,
		       map_t * map, noise_t * noise, fe_t * fe);
__global__
void lb_collision_mrt2(kernel_ctxt_t * ktx, lb_t * lb, hydro_t * hydro,
		       fe_symm_t * fe, noise_t * noise);

int lb_collision_mrt(lb_t * lb, hydro_t * hydro, map_t * map,
		     noise_t * noise, fe_t * fe);
int lb_collision_binary(lb_t * lb, hydro_t * hydro, noise_t * noise,
			fe_symm_t * fe);

static __host__ __device__
void lb_collision_fluctuations(lb_t * lb, noise_t * noise, int index,
			       double shat[3][3], double ghat[NVEL]);
int lb_collision_noise_var_set(lb_t * lb, noise_t * noise);
static __host__ int lb_collision_parameters_commit(lb_t * lb);

static __device__
void lb_collision_mrt1_site(lb_t * lb, hydro_t * hydro, map_t * map,
			    noise_t * noise, fe_t * fe, const int index0);
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

  if (ndist == 1) lb_collision_mrt(lb, hydro, map, noise, fe);
  if (ndist == 2) lb_collision_binary(lb, hydro, noise, (fe_symm_t *) fe);

  return 0;
}

/*****************************************************************************
 *
 *  lb_collision_mrt_site
 *
 *  Single fluid collision driver (multiple relaxation time).
 *
 *****************************************************************************/

__host__ int lb_collision_mrt(lb_t * lb, hydro_t * hydro, map_t * map,
			      noise_t * noise, fe_t * fe) {
  int nlocal[3];
  dim3 nblk, ntpb;
  fe_t * fetarget = NULL;
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
  if (fe) fe->func->target(fe, &fetarget);

  TIMER_start(TIMER_COLLIDE_KERNEL);

  tdpLaunchKernel(lb_collision_mrt1, nblk, ntpb, 0, 0, ctxt->target,
		  lb->target, hydro->target, map->target, noise->target,
		  fetarget);

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
		       map_t * map, noise_t * noise, fe_t * fe) {
  int kindex;
  int kiter;

  kiter = kernel_vector_iterations(ktx);

  for_simt_parallel(kindex, kiter, NSIMDVL) {
    int index0;
    index0 = kernel_baseindex(ktx, kindex);
    lb_collision_mrt1_site(lb, hydro, map, noise, fe, index0);
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
			    noise_t * noise, fe_t * fe, const int index0) {
  
  int p, m;                               /* velocity index */
  int ia, ib;                             /* indices ("alphabeta") */
  int iv=0;                               /* SIMD loop counter */
  double mode[NVEL*NSIMDVL];              /* Modes; hydrodynamic + ghost */
  double rho[NSIMDVL], rrho[NSIMDVL];     /* Density, reciprocal density */
  double u[3][NSIMDVL] = {0};             /* Velocity */
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

  for_simd_v(iv, NSIMDVL) includeSite[iv] = 1;

  for_simd_v(iv, NSIMDVL) {
    if (map->status[index0+iv] != MAP_FLUID) {
      includeSite[iv] = 0;
      fullchunk = 0;
    }
  }

  /* Default to fluctuations off; shat, ghat are zero */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL) shat[ia][ib][iv] = 0.0;
    }
  }

  for (ia = 0; ia < NVEL; ia++) {
    for_simd_v(iv, NSIMDVL) ghat[ia][iv] = 0.0;
  }

  /* Load SIMD vectors for distribution and force */

  for (p = 0; p < NVEL; p++) {
    for_simd_v(iv, NSIMDVL) fchunk[p*NSIMDVL+iv] = 
      lb->f[ LB_ADDR(_lbp.nsite, 1, NVEL, index0 + iv, LB_RHO, p) ];
  }

  for (ia = 0; ia < 3; ia++) {
    for_simd_v(iv, NSIMDVL) {
      force[ia][iv] = _cp.force_global[ia] 
	+ hydro->f[addr_rank1(hydro->nsite, NHDIM, index0+iv, ia)];
    }
  }
  
  /* Compute all the modes */

#ifdef _D3Q19_
    d3q19_f2mode_chunk(mode, fchunk);
#else
    for (m = 0; m < NVEL; m++) {
      for_simd_v(iv, NSIMDVL) mode[m*NSIMDVL+iv] = 0.0;
      for (p = 0; p < NVEL; p++) {
	for_simd_v(iv, NSIMDVL) {
	  mode[m*NSIMDVL+iv] += fchunk[p*NSIMDVL+iv]*_lbp.ma[m][p];
	}
      }
    }
#endif

  /* For convenience, write out the physical modes, that is,
   * rho, NDIM components of velocity, independent components
   * of stress (upper triangle), and lower triangle. */

  for_simd_v(iv, NSIMDVL) rho[iv] = mode[0*NSIMDVL+iv];
  for (ia = 0; ia < NDIM; ia++) {
    for_simd_v(iv, NSIMDVL) u[ia][iv] = mode[(1 + ia)*NSIMDVL+iv];
  }

  m = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      for_simd_v(iv, NSIMDVL) s[ia][ib][iv] = mode[(1 + NDIM + m)*NSIMDVL+iv];
      m++;
    }
  }
    
  for (ia = 1; ia < NDIM; ia++) {
    for (ib = 0; ib < ia; ib++) {
      for_simd_v(iv, NSIMDVL) s[ia][ib][iv] = s[ib][ia][iv];
    }
  }

  /* Compute the local velocity, taking account of any body force */
    
  for_simd_v(iv, NSIMDVL) rrho[iv] = 1.0/rho[iv];

  for (ia = 0; ia < NDIM; ia++) {      
    for_simd_v(iv, NSIMDVL) {
      u[ia][iv] = rrho[iv]*(u[ia][iv] + 0.5*force[ia][iv]);  
    }
  }
   
  /* Relax stress with different shear and bulk viscosity */

  for_simd_v(iv, NSIMDVL) {
    tr_s[iv]   = 0.0;
    tr_seq[iv] = 0.0;
  }

  if (fe && fe->use_stress_relaxation) {
    double symm[3][3][NSIMDVL];

    fe->func->str_symm_v(fe, index0, symm);

    for (ia = 0; ia < NDIM; ia++) {
      /* Set equilibrium stress */
      for (ib = 0; ib < NDIM; ib++) {
	for_simd_v(iv, NSIMDVL) {
	  seq[ia][ib][iv] = rho[iv]*u[ia][iv]*u[ib][iv] + symm[ia][ib][iv];
	}
      }
      /* Compute trace */
      for_simd_v(iv, NSIMDVL){
	tr_s[iv]   += s[ia][ia][iv];
	tr_seq[iv] += seq[ia][ia][iv];
      }
    }
  }
  else {

    for (ia = 0; ia < NDIM; ia++) {
      /* Set equilibrium stress */
      for (ib = 0; ib < NDIM; ib++) {
	for_simd_v(iv, NSIMDVL) {
	  seq[ia][ib][iv] = rho[iv]*u[ia][iv]*u[ib][iv];
	}
      }
      /* Compute trace */
      for_simd_v(iv, NSIMDVL){
	tr_s[iv]   += s[ia][ia][iv];
	tr_seq[iv] += seq[ia][ia][iv];
      }
    }
  }
    
  /* Form traceless parts */
  for (ia = 0; ia < NDIM; ia++) {
    for_simd_v(iv, NSIMDVL){
      s[ia][ia][iv]   -= rdim*tr_s[iv];
      seq[ia][ia][iv] -= rdim*tr_seq[iv];
    }
  }
    
  /* Relax each mode */
  for_simd_v(iv, NSIMDVL) {
    tr_s[iv] = tr_s[iv] - _lbp.rtau[LB_TAU_BULK]*(tr_s[iv] - tr_seq[iv]);
  }

  for (ia = 0; ia < NDIM; ia++) {
    for (ib = 0; ib < NDIM; ib++) {
      for_simd_v(iv, NSIMDVL) {
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
    for_simd_v(iv, NSIMDVL) mode[(1 + ia)*NSIMDVL+iv] += force[ia][iv];
  }

  m = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      for_simd_v(iv, NSIMDVL) {
	mode[(1 + NDIM + m)*NSIMDVL+iv] = s[ia][ib][iv] + shat[ia][ib][iv];
      }
      m++;
    }
  }

  /* Ghost modes are relaxed toward zero equilibrium. */

  for (m = NHYDRO; m < NVEL; m++) {  
    for_simd_v(iv, NSIMDVL) {
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
      for_simd_v(iv, NSIMDVL) ftmp[iv] = 0.0;
      for (m = 0; m < NVEL; m++) {
	for_simd_v(iv, NSIMDVL) ftmp[iv] += _lbp.mi[p][m]*mode[m*NSIMDVL+iv];
      }
      for_simd_v(iv, NSIMDVL) fchunk[p*NSIMDVL+iv] = ftmp[iv];
    }
#endif

  /* Write SIMD chunks back to main arrays. */

  if (fullchunk) {
    /* distribution */
    for (p = 0; p < NVEL; p++) {
      for_simd_v(iv, NSIMDVL) { 
	lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0+iv, LB_RHO, p)] = fchunk[p*NSIMDVL+iv];
      }
    }
    /* velocity */
    for (ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	hydro->u[addr_rank1(hydro->nsite, NHDIM, index0+iv, ia)] = u[ia][iv];
      }
    }
  }
  else {
    for_simd_v(iv, NSIMDVL) {
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

  for_simt_parallel(kindex, kiter, NSIMDVL) {
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
      for_simd_v(iv, NSIMDVL) shat[ia][ib][iv] = 0.0;
    }
  }

  for (m = NHYDRO; m < NVEL; m++) {
    for_simd_v(iv, NSIMDVL) ghat[m][iv] = 0.0;
  }


#ifdef _D3Q19_
  for (p = 0; p < NVEL; p++) {
    for_simd_v(iv, NSIMDVL) {
      f[p*NSIMDVL+iv]
	= lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0 + iv, LB_RHO, p)];
    }
  }
  d3q19_f2mode_chunk(mode, f);
#else
  /* Compute all the modes */
  for (m = 0; m < NVEL; m++) {
    for_simd_v(iv, NSIMDVL) {
      mode[m*NSIMDVL+iv] = 0.0;
    }
    for (p = 0; p < NVEL; p++) {
      for_simd_v(iv, NSIMDVL) {
	mode[m*NSIMDVL+iv] += _lbp.ma[m][p]
	  *lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0 + iv, LB_RHO, p)];
      }
    }
  }
#endif

  /* For convenience, write out the physical modes. */
  
  for_simd_v(iv, NSIMDVL) rho[iv] = mode[0*NSIMDVL+iv];
  for (ia = 0; ia < 3; ia++) {
    for_simd_v(iv, NSIMDVL) u[ia][iv] = mode[(1 + ia)*NSIMDVL+iv];
  }

  m = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      for_simd_v(iv, NSIMDVL) s[ia][ib][iv] = mode[(1 + NDIM + m)*NSIMDVL+iv];
      m++;
    }
  }

  for (ia = 1; ia < NDIM; ia++) {
    for (ib = 0; ib < ia; ib++) {
      for_simd_v(iv, NSIMDVL) s[ia][ib][iv] = s[ib][ia][iv];
    }
  }

  /* Compute the local velocity, taking account of any body force */
  
  for_simd_v(iv, NSIMDVL) rrho[iv] = 1.0/rho[iv];
  
  for (ia = 0; ia < 3; ia++) {
    for_simd_v(iv, NSIMDVL) {
      force[ia][iv] = _cp.force_global[ia] 
	+ hydro->f[addr_rank1(hydro->nsite, NHDIM, index0+iv, ia)];
      u[ia][iv] = rrho[iv]*(u[ia][iv] + 0.5*force[ia][iv]);  
    }
  }

  for (ia = 0; ia < 3; ia++) {   
    for_simd_v(iv, NSIMDVL) {
      hydro->u[addr_rank1(hydro->nsite, NHDIM, index0+iv, ia)] = u[ia][iv];
    }
  }
  
  /* Compute the thermodynamic component of the stress */

  fe_symm_str_v(fe, index0, sth);

  /* Relax stress with different shear and bulk viscosity */
  
  for_simd_v(iv, NSIMDVL) {
    tr_s[iv]   = 0.0;
    tr_seq[iv] = 0.0;
  }
  
  for (ia = 0; ia < 3; ia++) {
    /* Set equilibrium stress, which includes thermodynamic part */
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL) {
	seq[ia][ib][iv] = rho[iv]*u[ia][iv]*u[ib][iv] + sth[ia][ib][iv];
      }
    }
    /* Compute trace */
    for_simd_v(iv, NSIMDVL) {
      tr_s[iv]   += s[ia][ia][iv];
      tr_seq[iv] += seq[ia][ia][iv];
    }
  }
  
  /* Form traceless parts */
  for (ia = 0; ia < 3; ia++) {
    for_simd_v(iv, NSIMDVL) {
      s[ia][ia][iv]   -= r3*tr_s[iv];
      seq[ia][ia][iv] -= r3*tr_seq[iv];
    }
  }

  
  /* Relax each mode */
  for_simd_v(iv, NSIMDVL)
    tr_s[iv] = tr_s[iv] - _lbp.rtau[LB_TAU_BULK]*(tr_s[iv] - tr_seq[iv]);
  
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      for_simd_v(iv, NSIMDVL) {
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
    for_simd_v(iv, NSIMDVL) mode[(1 + ia)*NSIMDVL+iv] += force[ia][iv];
  }

  m = 0;
  for (ia = 0; ia < NDIM; ia++) {
    for (ib = ia; ib < NDIM; ib++) {
      for_simd_v(iv, NSIMDVL) {
	mode[(1 + NDIM + m)*NSIMDVL+iv] = s[ia][ib][iv] + shat[ia][ib][iv];
      }
      m++;
    }
  }

  /* Ghost modes are relaxed toward zero equilibrium. */

  for (m = NHYDRO; m < NVEL; m++) { 
    for_simd_v(iv, NSIMDVL)  {
      mode[m*NSIMDVL+iv] = mode[m*NSIMDVL+iv] 
	- lb->param->rtau[m]*(mode[m*NSIMDVL+iv] - 0.0) + ghat[m][iv];
    }
  }

  /* Project post-collision modes back onto the distribution */

#ifdef _D3Q19_  
  d3q19_mode2f_chunk(mode, f);
  for (p = 0; p < NVEL; p++) {
    for_simd_v(iv, NSIMDVL) {
      lb->f[LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0 + iv, LB_RHO, p)] =
	f[p*NSIMDVL+iv];
    }
  }
#else    
  for (p = 0; p < NVEL; p++) {
    for_simd_v(iv, NSIMDVL) f[p*NSIMDVL+iv] = 0.0;
    for (m = 0; m < NVEL; m++) {
      for_simd_v(iv, NSIMDVL) f[p*NSIMDVL+iv] += _lbp.mi[p][m]*mode[m*NSIMDVL+iv];
    }
    for_simd_v(iv, NSIMDVL) {
      lb->f[LB_ADDR(_lbp.nsite, NDIST, NVEL, index0+iv, LB_RHO, p)] = f[p*NSIMDVL+iv];
    }
  }
#endif

  /* Now, the order parameter distribution */
  for_simd_v(iv, NSIMDVL) {
    phi[iv] = fe->phi->data[addr_rank0(fe->phi->nsites, index0 + iv)];
  }

  for_simd_v(iv, NSIMDVL) {
    fe_symm_mu(fe, index0 + iv, mu + iv);
    jphi[X][iv] = 0.0;
    jphi[Y][iv] = 0.0;
    jphi[Z][iv] = 0.0;
  }

  for (p = 1; p < NVEL; p++) {
    for (ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) {
	jphi[ia][iv] += _lbp.cv[p][ia]* 
	lb->f[ LB_ADDR(_lbp.nsite, _lbp.ndist, NVEL, index0+iv, LB_PHI, p) ];
      }
    }
  }

  /* Relax order parameter modes. See the comments above. */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      for_simd_v(iv, NSIMDVL) { 
	sphi[ia][ib][iv] = phi[iv]*u[ia][iv]*u[ib][iv] + mu[iv]*d[ia][ib];
        /* sphi[ia][ib] = phi*u[ia]*u[ib] + cs2*mobility*mu*d_[ia][ib];*/
      }
    }
    for_simd_v(iv, NSIMDVL) {
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

    for_simd_v(iv, NSIMDVL) {
      jdotc[iv]    = 0.0;
      sphidotq[iv] = 0.0;
    }
    
    for (ia = 0; ia < 3; ia++) {
      for_simd_v(iv, NSIMDVL) jdotc[iv] += jphi[ia][iv]*_lbp.cv[p][ia];
      for (ib = 0; ib < 3; ib++) {
	for_simd_v(iv, NSIMDVL) sphidotq[iv] += sphi[ia][ib][iv]*_lbp.q[p][ia][ib];
      }
    }
    
    /* Project all this back to the distributions. The magic
     * here is to move phi into the non-propagating distribution. */
    for_simd_v(iv, NSIMDVL) { 
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

  int ia;
  double t;
  double force_constant[3];
  double fpulse_frequency;
  double fpulse_frequency_rad;
  double fpulse_amplitude[3] = {0.0, 0.0, 0.0};
  double force_pulsatile[3] = {0.0, 0.0, 0.0};

  PI_DOUBLE(pi);

  assert(lb);

  physics_ref(&phys);
  physics_fbody(phys, force_constant);

  /* Pulse force (time dependent) */
  physics_fpulse(phys, fpulse_amplitude);
  physics_fpulse_frequency(phys, &fpulse_frequency);

  t = physics_control_timestep(phys);
  fpulse_frequency_rad =  2.0*pi*fpulse_frequency; 

  for (ia = 0; ia < 3; ia++) {
    force_pulsatile[ia] = fpulse_amplitude[ia]*sin(fpulse_frequency_rad*t);
    p.force_global[ia] = force_constant[ia]+force_pulsatile[ia];
  }

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

  int m, iv;

   for (m = 0; m < NVEL; m++) { 
       for_simd_v(iv, NSIMDVL) mode[m*NSIMDVL+iv] = 0.0; 
   }


  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[0*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c1;

  /* m=1*/
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[1*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*-c1;

  /* m=2*/
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv]=0.;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[2*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*-c1;

  /* m=3*/
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[3*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c0;

  /* m=4*/
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[4*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*t3;

  /* m=5*/
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[5*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c1;

  /* m=6*/
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[6*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c0;

  /* m=7*/
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[7*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*t3;

  /* m=8*/
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[8*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c0;

  /* m=9*/
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*t3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*-r3;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*t3 ;
  for_simd_v(iv, NSIMDVL) mode[9*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*-r3;

  /* m=10*/
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[10*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*-c2;

  /* m=11*/
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c2;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[11*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c2;

  /* m=12*/
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c2;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[12*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c2;

  /* m=13*/
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c2;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[13*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c0;

  /* m=14*/
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[14*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c0;

  /* m=15*/
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[15*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c0;

  /* m=16*/
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[16*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c0;

  /* m=17*/
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*-c1;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*c0;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[17*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c0;

  /* m=18*/
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[0*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[1*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[2*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[3*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[4*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[5*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[6*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[7*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[8*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[9*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[10*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[11*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[12*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[13*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[14*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[15*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[16*NSIMDVL+iv]*-c2;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[17*NSIMDVL+iv]*c1;
  for_simd_v(iv, NSIMDVL) mode[18*NSIMDVL+iv] += fchunk[18*NSIMDVL+iv]*c1;
 
}

__device__ void d3q19_mode2f_chunk(double* mode, double* fchunk) {

  double ftmp[NSIMDVL];

  int iv;

  
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w0*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r2*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r2*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r2*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL)   fchunk[0*NSIMDVL+iv] = ftmp[iv];
  
  /* p=1*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r4*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[1*NSIMDVL+iv] = ftmp[iv];
  
  /* p=2*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r4*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wd*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -we*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[2*NSIMDVL+iv] =  ftmp[iv];
  
  /* p=3*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w1*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r4*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -w1*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[3*NSIMDVL+iv] =  ftmp[iv];
  
  /* p=4*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r4*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wd*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -we*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[4*NSIMDVL+iv] = ftmp[iv];
  
  /* p=5 */
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r4*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[5*NSIMDVL+iv] = ftmp[iv];
  
  /* p=6 */
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r4*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wd*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += we*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[6*NSIMDVL+iv] =  ftmp[iv];
  
  /* p=7 */
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w1*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r4*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -w1*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[7*NSIMDVL+iv] = ftmp[iv];
  
  /* p=8 */
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r4*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wd*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += we*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[8*NSIMDVL+iv] =  ftmp[iv];
  
  /* p=9*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w1*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r6*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -w1*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[9*NSIMDVL+iv] = ftmp[iv];
  
  /* p=10*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w1*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r6*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -w1*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[10*NSIMDVL+iv] = ftmp[iv];
  
  /* p=11 */
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r4*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wd*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += we*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[11*NSIMDVL+iv] =  ftmp[iv];
  
  /* p=12*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w1*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r6*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r4*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -w1*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[12*NSIMDVL+iv] = ftmp[iv];
  
  /* p=13 */
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r4*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wd*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += we*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[13*NSIMDVL+iv] =  ftmp[iv];
  
  /* p=14 */
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r4*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[14*NSIMDVL+iv] =  ftmp[iv];
  
  /* p=15*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r4*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wd*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -we*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r8*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[15*NSIMDVL+iv] =  ftmp[iv];
  
  /* p=16*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w1*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r6*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r6*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wb*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -r4*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -w1*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[16*NSIMDVL+iv] =ftmp[iv];
  
  /* p=17*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r4*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wd*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -we*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r8*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[17*NSIMDVL+iv] = ftmp[iv];
  
  /* p=18*/
  for_simd_v(iv, NSIMDVL) ftmp[iv]=0.;
  for_simd_v(iv, NSIMDVL) ftmp[iv] += w2*mode[0*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[1*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wa*mode[2*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[3*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[4*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += r4*mode[5*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[6*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[7*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[8*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[9*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += -wb*mode[10*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[11*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wa*mode[12*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[13*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[14*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[15*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[16*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += c0*mode[17*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) ftmp[iv] += wc*mode[18*NSIMDVL+iv];
  for_simd_v(iv, NSIMDVL) fchunk[18*NSIMDVL+iv] = ftmp[iv];


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

  for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

  for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 0) ] 
        = w0*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4) + phi[iv];


  for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][1][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][0][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 1) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][2][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][0][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 2) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 3) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][2][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][0][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 4) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][1][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][0][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 5) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][2][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][1][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 6) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 7) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][2][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][1][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 8) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 9) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 10) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][2][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][1][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 11) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 12) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][2][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][1][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 13) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][1][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][0][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 14) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] += jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][2][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][0][iv]*-1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 15) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 16) ] 
        = w1*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Z][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][2][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*-3.3333333333333331e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][0][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*6.6666666666666663e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 17) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);


 for_simd_v(iv, NSIMDVL) { jdotc[iv]    = 0.0; sphidotq[iv] = 0.0;} 

  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[X][iv];
  for_simd_v(iv, NSIMDVL)  jdotc[iv] -= jphi[Y][iv];
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][0][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[0][1][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][0][iv]*1.0000000000000000e+00;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[1][1][iv]*6.6666666666666663e-01;
  for_simd_v(iv, NSIMDVL)  sphidotq[iv] += sphi[2][2][iv]*-3.3333333333333331e-01;

 for_simd_v(iv, NSIMDVL) 
     f[ LB_ADDR(_lbp.nsite, NDIST, NVEL, baseIndex+iv, LB_PHI, 18) ] 
        = w2*(jdotc[iv]*rcs2 + sphidotq[iv]*r2rcs4);

  return;
}
#endif
