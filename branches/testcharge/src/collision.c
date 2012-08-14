/*****************************************************************************
 *
 *  collision.c
 *
 *  Collision stage routines and associated data.
 *
 *  Isothermal fluctuations following Adhikari et al., Europhys. Lett
 *  (2005).
 *
 *  The relaxation times can be set to give either 'm10', BGK, or
 *  'two-relaxation' time (TRT) models.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "physics.h"
#include "model.h"
#include "site_map.h"
#include "collision.h"
#include "fluctuations.h"

#include "free_energy.h"
#include "phi_cahn_hilliard.h"

#include "control.h"
#include "propagation_ode.h"

static int nmodes_ = NVEL;               /* Modes to use in collsion stage */
static int nrelax_ = RELAXATION_M10;     /* Default is m10 */
static int isothermal_fluctuations_ = 0; /* Flag for noise. */

static double rtau_shear;       /* Inverse relaxation time for shear modes */
static double rtau_bulk;        /* Inverse relaxation time for bulk modes */
static double var_shear;        /* Variance for shear mode fluctuations */
static double var_bulk;         /* Variance for bulk mode fluctuations */
static double rtau_[NVEL];      /* Inverse relaxation times */
static double noise_var[NVEL];  /* Noise variances */

static fluctuations_t * fl_;

static int collision_multirelaxation(hydro_t * hydro);
static int collision_binary_lb(hydro_t * hydro);

static void fluctuations_off(double shat[3][3], double ghat[NVEL]);
       void collision_fluctuations(int index, double shat[3][3],
				   double ghat[NVEL]);

/*****************************************************************************
 *
 *  collide
 *
 *  Driver routine for the collision stage.
 *
 *  Note that the ODE propagation currently uses ndist == 2, as
 *  well as the LB binary, hence the logic.
 *
 *  We allow hydro to be NULL, in which case there is no hydrodynamics!
 * 
 *****************************************************************************/

int collide(hydro_t * hydro) {

  int ndist;

  if (hydro == NULL) return 0;

  ndist = distribution_ndist();
  collision_relaxation_times_set();

  if (ndist == 1 || is_propagation_ode() == 1) collision_multirelaxation(hydro);
  if (ndist == 2 && is_propagation_ode() == 0) collision_binary_lb(hydro);

  return 0;
}

/*****************************************************************************
 *
 *  collision_multirelaxation
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

int collision_multirelaxation(hydro_t * hydro) {

  int       N[3];
  int       ic, jc, kc, index;       /* site indices */
  int       p, m;                    /* velocity index */
  int       ia, ib;                  /* indices ("alphabeta") */
  int       ndist;

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
  double    f[NVEL];

  assert(hydro);

  ndist = distribution_ndist();
  coords_nlocal(N);
  fluctuations_off(shat, ghat);
  fluid_body_force(force_global);

  rdim = 1.0/NDIM;

  for (ia = 0; ia < 3; ia++) {
    u[ia] = 0.0;
  }

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = coords_index(ic, jc, kc);

	/* Compute all the modes */

	distribution_index(index, 0, f);

	for (m = 0; m < nmodes_; m++) {
	  mode[m] = 0.0;
	  for (p = 0; p < NVEL; p++) {
	    mode[m] += f[p]*ma_[m][p];
	  }
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

	if (isothermal_fluctuations_) {
	  collision_fluctuations(index, shat, ghat);
	}

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

	/* Project post-collision modes back onto the distribution */

	for (p = 0; p < NVEL; p++) {
	  f[p] = 0.0;
	  for (m = 0; m < nmodes_; m++) {
	    f[p] += mi_[p][m]*mode[m];
	  }
	}

	distribution_index_set(index, 0, f);

	/* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  collision_binary_lb
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

int collision_binary_lb(hydro_t * hydro) {

  int       N[3];
  int       ic, jc, kc, index;       /* site indices */
  int       p, m;                    /* velocity index */
  int       i, j;                    /* summed over indices ("alphabeta") */
  int       ndist;

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
  double    force_global[3];
  double    f[NVEL];

  const double   r3     = (1.0/3.0);


  double    phi, jdotc, sphidotq;    /* modes */
  double    jphi[3];
  double    sth[3][3], sphi[3][3];
  double    mu;                      /* Chemical potential */
  double    rtau2;
  double    mobility;
  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  double (* chemical_potential)(const int index, const int nop);
  void   (* chemical_stress)(const int index, double s[3][3]);

  assert (NDIM == 3);
  assert(hydro);

  ndist = distribution_ndist();
  coords_nlocal(N);
  fluid_body_force(force_global);

  chemical_potential = fe_chemical_potential_function();
  chemical_stress = fe_chemical_stress_function();

  mobility = phi_cahn_hilliard_mobility();
  rtau2 = 2.0 / (1.0 + 6.0*mobility);
  fluctuations_off(shat, ghat);

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = coords_index(ic, jc, kc);

	/* Compute all the modes */

	distribution_index(index, 0, f);

	for (m = 0; m < nmodes_; m++) {
	  mode[m] = 0.0;
	  for (p = 0; p < NVEL; p++) {
	    mode[m] += f[p]*ma_[m][p];
	  }
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
	hydro_f_local(hydro, index, force_local);

	for (i = 0; i < 3; i++) {
	  force[i] = (force_global[i] + force_local[i]);
	  u[i] = rrho*(u[i] + 0.5*force[i]);  
	}
	hydro_u_set(hydro, index, u);

	/* Compute the thermodynamic component of the stress */

	chemical_stress(index, sth);

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
	tr_s = tr_s - rtau_bulk*(tr_s - tr_seq);

	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    s[i][j] -= rtau_shear*(s[i][j] - seq[i][j]);
	    s[i][j] += d_[i][j]*r3*tr_s;

	    /* Correction from body force (assumes equal relaxation times) */

	    s[i][j] += (2.0-rtau_shear)*(u[i]*force[j] + force[i]*u[j]);
	    shat[i][j] = 0.0;
	  }
	}

	if (isothermal_fluctuations_) {
	  collision_fluctuations(index, shat, ghat);
	}

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

	for (m = NHYDRO; m < nmodes_; m++) {
	  mode[m] = mode[m] - rtau_[m]*(mode[m] - 0.0) + ghat[m];
	}

	/* Project post-collision modes back onto the distribution */

	for (p = 0; p < NVEL; p++) {
	  f[p] = 0.0;
	  for (m = 0; m < nmodes_; m++) {
	    f[p] += mi_[p][m]*mode[m];
	  }
	}

	distribution_index_set(index, 0, f);

	/* Now, the order parameter distribution */

	mu = chemical_potential(index, 0);
	distribution_index(index, 1, f);

	phi = f[0];
	jphi[X] = 0.0;
	jphi[Y] = 0.0;
	jphi[Z] = 0.0;
	for (p = 1; p < NVEL; p++) {
	  phi += f[p];
	  for (i = 0; i < 3; i++) {
	    jphi[i] += f[p]*cv[p][i];
	  }
	}

	/* if (kc == 1) printf("%3d %3d %14.7e %14.7e\n", ic, jc, phi, mu);*/

	/* Relax order parameters modes. See the comments above. */

	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    sphi[i][j] = phi*u[i]*u[j] + mu*d_[i][j];
	    /* sphi[i][j] = phi*u[i]*u[j] + mobility*mu*d_[i][j];*/
	  }
	  jphi[i] = jphi[i] - rtau2*(jphi[i] - phi*u[i]);
	  /* jphi[i] = phi*u[i];*/
	}

	/* Now update the distribution */

	for (p = 0; p < NVEL; p++) {

	  int dp0 = (p == 0);
	  jdotc    = 0.0;
	  sphidotq = 0.0;

	  for (i = 0; i < 3; i++) {
	    jdotc += jphi[i]*cv[p][i];
	    for (j = 0; j < 3; j++) {
	      sphidotq += sphi[i][j]*q_[p][i][j];
	    }
	  }

	  /* Project all this back to the distributions. The magic
	   * here is to move phi into the non-propagating distribution. */

	  f[p] = wv[p]*(jdotc*rcs2 + sphidotq*r2rcs4) + phi*dp0;
	}

	distribution_index_set(index, 1, f);

	/* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fluctuations_off
 *
 *  Return zero fluctuations for stress (shat) and ghost (ghat) modes.
 *
 *****************************************************************************/

static void fluctuations_off(double shat[3][3], double ghat[NVEL]) {

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

/*****************************************************************************
 *
 *  test_isothermal_fluctuations
 *
 *  Reports the equipartition of momentum, and the actual temperature
 *  cf. the expected (input) temperature.
 *
 *****************************************************************************/

void test_isothermal_fluctuations(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n;

  double glocal[4];
  double gtotal[4];
  double rrho;
  double gsite[3];

  if (isothermal_fluctuations_ == 0) return;

  coords_nlocal(nlocal);

  glocal[X] = 0.0;
  glocal[Y] = 0.0;
  glocal[Z] = 0.0;
  glocal[3] = 0.0; /* volume of fluid */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	if (site_map_get_status_index(index) != FLUID) continue;

	rrho = 1.0/distribution_zeroth_moment(index, 0);
	distribution_first_moment(index, 0, gsite);

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
  info("[measd/kT] %14.7e %14.7e\n", gtotal[X] + gtotal[Y] + gtotal[Z],
       get_kT()*NDIM);

  return;
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
 *  collision_fluctuations_on
 *
 *****************************************************************************/

void collision_fluctuations_on(void) {

  isothermal_fluctuations_ = 1;

  return;
}

/*****************************************************************************
 *
 *  collision_fluctuations_off
 *
 *****************************************************************************/

void collision_fluctuations_off(void) {

  isothermal_fluctuations_ = 0;

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
 *  collision_relaxation_times_set
 *
 *  Note there is an extra normalisation in the lattice fluctuations
 *  which would otherwise give effective kT = cs2
 *
 *****************************************************************************/

void collision_relaxation_times_set(void) {

  int p;
  double kt;
  double tau_s;
  double tau_b;
  double tau_g;

  double dt_ode=1.0;
  extern int is_propagation_ode(void);
 
  if (is_propagation_ode()) dt_ode = propagation_ode_get_tstep();

  /* Initialise the relaxation times */

  rtau_shear = 2.0 / (1.0 + 6.0*get_eta_shear()/dt_ode);
  rtau_bulk  = 2.0 / (1.0 + 6.0*get_eta_bulk()/dt_ode);

  if (nrelax_ == RELAXATION_M10) {
    for (p = NHYDRO; p < NVEL; p++) {
      rtau_[p] = 1.0;
    }
  }

  if (nrelax_ == RELAXATION_BGK) {
    for (p = NHYDRO; p < NVEL; p++) {
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

  /* Density and momentum */

  tau[0] = 0.0;

  for (ia = 0; ia < NDIM; ia++) {
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
 *  collision_init
 *
 *  Set up the noise generator.
 *
 *****************************************************************************/

void collision_init(void) {

  int nsites;
  int ic, jc, kc, index;
  int is_local;
  int nlocal[3];
  int noffset[3];
  int ntotal[3];

  unsigned int serial[4] = {13, 829, 2441, 22383979};
  unsigned int state[4];

  ntotal[X] = N_total(X);
  ntotal[Y] = N_total(Y);
  ntotal[Z] = N_total(Z);

  nsites = coords_nsites();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  /* This is slightly incestuous; we are going to use the uniform
   * generator from fluctuations to generate, in serial, the
   * initial states for the fluctuation generator. The initialisation
   * of the serial state is absolutely fixed, as above. */

  fl_ = fluctuations_create(nsites);

  for (ic = 1; ic <= ntotal[X]; ic++) {
    for (jc = 1; jc <= ntotal[Y]; jc++) {
      for (kc = 1; kc <= ntotal[Z]; kc++) {
	state[0] = fluctuations_uniform(serial);
	state[1] = fluctuations_uniform(serial);
	state[2] = fluctuations_uniform(serial);
	state[3] = fluctuations_uniform(serial);
	is_local = 1;
	if (ic <= noffset[X] || ic > noffset[X] + nlocal[X]) is_local = 0;
	if (jc <= noffset[Y] || jc > noffset[Y] + nlocal[Y]) is_local = 0;
	if (kc <= noffset[Z] || kc > noffset[Z] + nlocal[Z]) is_local = 0;
	if (is_local) {
	  index = coords_index(ic-noffset[X], jc-noffset[Y], kc-noffset[Z]);
	  fluctuations_state_set(fl_, index, state);
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  collision_finish
 *
 *****************************************************************************/

void collision_finish(void) {

  assert(fl_);
  fluctuations_destroy(fl_);

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
 *****************************************************************************/

void collision_fluctuations(int index, double shat[3][3], double ghat[NVEL]) {

  int ia;
  double tr;
  double random[NFLUCTUATION];
  const double r3 = (1.0/3.0);

  assert(fl_);
  assert(NFLUCTUATION >= NDIM*(NDIM+1)/2);
  assert(NFLUCTUATION >= (NVEL - NHYDRO));

  /* Set symetric random stress matrix (elements with unit variance) */

  fluctuations_reap(fl_, index, random);

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

  tr = r3*(shat[X][X] + shat[Y][Y] + shat[Z][Z]);
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
    fluctuations_reap(fl_, index, random);
    for (ia = NHYDRO; ia < nmodes_; ia++) {
      ghat[ia] = noise_var[ia]*random[ia - NHYDRO];
    }
  }

  return;
}
