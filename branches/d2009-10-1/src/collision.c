/*****************************************************************************
 *
 *  collision.c
 *
 *  Collision stage routines and associated data.
 *
 *  $Id: collision.c,v 1.21.4.1 2009-11-04 10:20:43 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h> /* Remove strcmp */

#include "pe.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
#include "physics.h"
#include "runtime.h"
#include "control.h"
#include "free_energy.h"
#include "phi.h"
#include "phi_gradients.h"
#include "phi_force.h"
#include "phi_cahn_hilliard.h"
#include "phi_lb_coupler.h"
#include "phi_stats.h"
#include "lattice.h"

#include "util.h"
#include "utilities.h"
#include "communicate.h"
#include "leesedwards.h"
#include "model.h"
#include "collision.h"

#include "site_map.h"
#include "io_harness.h"

extern Site * site;

/* Variables (not static) */

double    siteforce[3];

/* Variables concerned with the collision */

static int    nmodes_ = NVEL;  /* Modes to use in collsion stage */

static double  noise0 = 0.1;   /* Initial noise amplitude    */
static double rtau_shear;      /* Inverse relaxation time for shear modes */
static double rtau_bulk;       /* Inverse relaxation time for bulk modes */
static double rtau_ghost = 1.0;/* Inverse relaxation time for ghost modes */
static double var_shear;       /* Variance for shear mode fluctuations */
static double var_bulk;        /* Variance for bulk mode fluctuations */

static double noise_var[NVEL]; /* Noise variances */

void MODEL_collide_multirelaxation(void);
void MODEL_collide_binary_lb(void);

/*****************************************************************************
 *
 *  collide
 *
 *  Driver routine for the collision stage.
 *
 *****************************************************************************/

void collide() {

#ifdef _SINGLE_FLUID_

  /* This is single fluid collision stage. */
  MODEL_collide_multirelaxation();

#else

  /* This is the binary LB collision. First, compute order parameter
   * gradients, then collision stage. The order of these calls is
   * important. */

  TIMER_start(TIMER_PHI_GRADIENTS);

  phi_compute_phi_site();
  phi_halo();
  phi_gradients_compute();

  TIMER_stop(TIMER_PHI_GRADIENTS);

  if (phi_is_finite_difference()) {
    phi_force_calculation();
    MODEL_collide_multirelaxation();
    phi_cahn_hilliard();
  }
  else {
    MODEL_collide_binary_lb();
  }

#endif

  return;
}

/*****************************************************************************
 *
 *  MODEL_collide_multirelaxation
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

void MODEL_collide_multirelaxation() {

  int       N[3];
  int       ic, jc, kc, index;       /* site indices */
  int       p, m;                    /* velocity index */
  int       i, j;                    /* summed over indices ("alphabeta") */

  double    mode[NVEL];              /* Modes; hydrodynamic + ghost */
  double    rho, rrho;               /* Density, reciprocal density */
  double    u[3];                    /* Velocity */
  double    s[3][3];                 /* Stress */
  double    seq[3][3];               /* Equilibrium stress */
  double    shat[3][3];              /* random stress */

  double    force[3];                /* External force */
  double    tr_s, tr_seq;

  double    force_local[3];
  const double   r3     = (1.0/3.0);

  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = get_site_index(ic, jc, kc);

	/* Compute all the modes */

	for (m = 0; m < nmodes_; m++) {
	  mode[m] = 0.0;
	  for (p = 0; p < NVEL; p++) {
	    mode[m] += site[index].f[p]*ma_[m][p];
	  }
	}

	/* For convenience, write out the physical modes. */

	rho = mode[0];
	for (i = 0; i < ND; i++) {
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
	hydrodynamics_get_force_local(index, force_local);

	for (i = 0; i < 3; i++) {
	  force[i] = (siteforce[i] + force_local[i]);
	  u[i] = rrho*(u[i] + 0.5*force[i]);
	}
	hydrodynamics_set_velocity(index, u);

	/* Relax stress with different shear and bulk viscosity */

	tr_s   = 0.0;
	tr_seq = 0.0;

	for (i = 0; i < 3; i++) {
	  /* Set equilibrium stress */
	  for (j = 0; j < 3; j++) {
	    seq[i][j] = rho*u[i]*u[j];
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

#ifdef _NOISE_
	get_fluctuations_stress(shat);
#endif

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
	  mode[m] = mode[m] - rtau_ghost*(mode[m] - 0.0);
#ifdef _NOISE_
	  mode[m] += noise_var[m]*ran_parallel_gaussian();
#endif
	}

	/* Project post-collision modes back onto the distribution */

	for (p = 0; p < NVEL; p++) {
	  site[index].f[p] = 0.0;
	  for (m = 0; m < nmodes_; m++) {
	    site[index].f[p] += mi_[p][m]*mode[m];
	  }
	}

	/* Next site */
      }
    }
  }
 
 TIMER_stop(TIMER_COLLIDE);

  return;
}

/*****************************************************************************
 *
 *  MODEL_collide_binary_lb
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

void MODEL_collide_binary_lb() {

  int       N[3];
  int       ic, jc, kc, index;       /* site indices */
  int       p, m;                    /* velocity index */
  int       i, j;                    /* summed over indices ("alphabeta") */

  double    mode[NVEL];              /* Modes; hydrodynamic + ghost */
  double    rho, rrho;               /* Density, reciprocal density */
  double    u[3];                    /* Velocity */
  double    s[3][3];                 /* Stress */
  double    seq[3][3];               /* equilibrium stress */
  double    shat[3][3];              /* random stress */

  double    force[3];                /* External force */
  double    tr_s, tr_seq;

  double    force_local[3];
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

  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);

  chemical_potential = fe_chemical_potential_function();
  chemical_stress = fe_chemical_stress_function();

  mobility = phi_ch_get_mobility();
  rtau2 = 2.0 / (1.0 + 6.0*mobility);

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = get_site_index(ic, jc, kc);

	/* Compute all the modes */

	for (m = 0; m < nmodes_; m++) {
	  mode[m] = 0.0;
	  for (p = 0; p < NVEL; p++) {
	    mode[m] += site[index].f[p]*ma_[m][p];
	  }
	}

	/* For convenience, write out the physical modes. */

	rho = mode[0];
	for (i = 0; i < ND; i++) {
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
	hydrodynamics_get_force_local(index, force_local);

	for (i = 0; i < 3; i++) {
	  force[i] = (siteforce[i] + force_local[i]);
	  u[i] = rrho*(u[i] + 0.5*force[i]);  
	}
	hydrodynamics_set_velocity(index, u);

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

#ifdef _NOISE_
	get_fluctuations_stress(shat);
#endif

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
	  mode[m] = mode[m] - rtau_ghost*(mode[m] - 0.0);
#ifdef _NOISE_
	  mode[m] += noise_var[m]*ran_parallel_gaussian();
#endif
	}

	/* Project post-collision modes back onto the distribution */

	for (p = 0; p < NVEL; p++) {
	  site[index].f[p] = 0.0;
	  for (m = 0; m < nmodes_; m++) {
	    site[index].f[p] += mi_[p][m]*mode[m];
	  }
	}

	/* Now, the order parameter distribution */

	phi = phi_get_phi_site(index);
	mu = chemical_potential(index, 0);

	jphi[X] = 0.0;
	jphi[Y] = 0.0;
	jphi[Z] = 0.0;
	for (p = 1; p < NVEL; p++) {
	  for (i = 0; i < 3; i++) {
	    jphi[i] += site[index].g[p]*cv[p][i];
	  }
	}

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

	  site[index].g[p] = wv[p]*(jdotc*rcs2 + sphidotq*r2rcs4) + phi*dp0;
	}

	/* Next site */
      }
    }
  }

  TIMER_stop(TIMER_COLLIDE);

  return;
}

/*----------------------------------------------------------------------------*/
/*!
 * Initialise model (allocate buffers, initialise velocities, etc.)
 *
 *- \c Arguments: void
 *- \c Returns:   void
 *- \c Buffers:   sets .halo, .rho, .phi
 *- \c Version:   2.0b1
 *- \c Last \c updated: 01/03/2002 by JCD
 *- \c Authors:   P. Bladon and JC Desplat
 *- \c Note:      none
 */
/*----------------------------------------------------------------------------*/

void MODEL_init( void ) {

  int     i,j,k,ind;
  int     N[3];
  int     offset[3];
  double   phi;
  double   phi0, rho0;
  double   mobility;
  char     filename[FILENAME_MAX];

  rho0 = get_rho0();
  phi0 = get_phi0();

  get_N_local(N);
  get_N_offset(offset);

  /* Now setup the rest of the simulation */

  site_map_init();

  /* If you want to read porous media information here you need this. */

  i = RUN_get_string_parameter("porous_media_format", filename, FILENAME_MAX);
  if (strcmp(filename, "ASCII") == 0) {
    info("Expecting porous media file format in ascii\n");
    io_info_set_processor_dependent(io_info_site_map);
    io_info_set_format_ascii(io_info_site_map);
  }

  if (strcmp(filename, "BINARY") == 0) {
    info("Expecting porous media file format in binary\n");
    /* ...is the defualt */
  }

  /* Default is "status_only" */
  i = RUN_get_string_parameter("porous_media_type", filename, FILENAME_MAX);
  if (strcmp(filename, "status_with_h") == 0) {
    info("Expecting site data to include wetting parameter h\n");
    site_map_io_status_with_h();
  }

  i = RUN_get_string_parameter("porous_media_file", filename, FILENAME_MAX);
  if (i == 1) {
    io_read(filename, io_info_site_map); 
    site_map_halo();
    phi_gradients_set_solid();
  }

  /* Distributions */

  init_site();

  ind = RUN_get_string_parameter("reduced_halo", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "yes") == 0) {
    info("\nUsing reduced halos\n\n");
    distribution_halo_set_reduced();
  }

  /* Order parameter */

  ind = RUN_get_string_parameter("phi_finite_difference", filename,
				 FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "yes") == 0) {
    phi_set_finite_difference();
    info("Switching order parameter to finite difference\n");

    i = 1;
    RUN_get_int_parameter("finite_difference_upwind_order", &i);
    phi_ch_set_upwind_order(i);
  }
  else {
    info("Order parameter is via lattice Boltzmann\n");
  }

  phi_init();

  ind = RUN_get_string_parameter("phi_format", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "ASCII") == 0) {
    io_info_set_format_ascii(io_info_phi);
    info("Setting phi I/O format to ASCII\n");
  }

  hydrodynamics_init();
  
  ind = RUN_get_string_parameter("vel_format", filename, FILENAME_MAX);
  if (ind != 0 && strcmp(filename, "ASCII") == 0) {
    io_info_set_format_ascii(io_info_velocity_);
    info("Setting velocity I/O format to ASCII\n"); 
  }

  /*
   * A number of options are offered to start a simulation:
   * 1. Read distribution functions site from file, then simply calculate
   *    all other properties (rho/phi/gradients/velocities)
   * 6. set rho/phi/velocity to default values, automatically set etc.
   */

  RUN_get_double_parameter("noise", &noise0);

  /* Option 1: read distribution functions from file */

  ind = RUN_get_string_parameter("input_config", filename, FILENAME_MAX);

  if (ind != 0) {

    info("Re-starting simulation at step %d with data read from "
	 "config\nfile(s) %s\n", get_step(), filename);

    /* Read distribution functions - sets both */
    io_read(filename, io_info_distribution_);
  } 
  else {
      /* 
       * Provides same initial conditions for rho/phi regardless of the
       * decomposition. 
       */
      
      /* Initialise lattice with constant density */
      /* Initialise phi with initial value +- noise */

      for(i=1; i<=N_total(X); i++)
	for(j=1; j<=N_total(Y); j++)
	  for(k=1; k<=N_total(Z); k++) {

	    phi = phi0 + noise0*(ran_serial_uniform() - 0.5);

	    /* For computation with single fluid and no noise */
	    /* Only set values if within local box */
	    if((i>offset[X]) && (i<=offset[X] + N[X]) &&
	       (j>offset[Y]) && (j<=offset[Y] + N[Y]) &&
	       (k>offset[Z]) && (k<=offset[Z] + N[Z]))
	      {
		ind = get_site_index(i-offset[X], j-offset[Y], k-offset[Z]);
#ifdef _SINGLE_FLUID_
		set_rho(rho0, ind);
		set_phi(phi0, ind);
#else
		set_rho(rho0, ind); 
		set_phi(phi,  ind);
		phi_set_phi_site(ind, phi);
#endif
	      }
	  }
  }

  ind = RUN_get_string_parameter("phi_initialisation", filename,
				 FILENAME_MAX);

  if (ind != 0 && strcmp(filename, "block") == 0) {
    info("Initialisng phi as block\n");
    phi_init_block();
  }

  if (ind != 0 && strcmp(filename, "bath") == 0) {
    info("Initialising phi for bath\n");
    phi_init_bath();
  }

  if (ind != 0 && strcmp(filename, "drop") == 0) {
    info("Initialising droplet\n");
    /* Pending refactoring */
    /* phi_lb_init_drop(0.125*L(X), interfacial_width());*/
    assert(0);
  }

  ind = RUN_get_double_parameter("psi_b", &rho0);

  if (ind != 0) {
    phi_init_surfactant(rho0);
  }

  /* Order parameter mobility (probably to move) */

  ind = RUN_get_double_parameter("mobility", &mobility);
  info("\nOrder parameter mobility M: %f\n", mobility);
  phi_ch_set_mobility(mobility);

  if (nop_ == 2) {
    ind = RUN_get_double_parameter("mobility_psi", &mobility);
    info("\nSurfactant mobility M: %f\n", mobility);
    phi_ch_op_set_mobility(mobility, 1);
  }

  return;
}

/****************************************************************************
 *
 *  RAND_init_fluctuations
 *
 *  Set variances for fluctuating lattice Boltzmann.
 *  Issues
 *    The 'physical' temperature is taken from the input
 *    and used throughout the code.
 *    Note there is an extra normalisation in the lattice fluctuations
 *    which would otherwise give effective kT = cs2
 *
 *    IMPORTANT: Note that the noise generation is not decomposition-
 *               independent when run in parallel.
 *
 ****************************************************************************/

void RAND_init_fluctuations() {

  int  p;
  char tmp[128];
  double tau_s, tau_b, tau_g, kt;

  p = RUN_get_double_parameter("temperature", &kt);
  set_kT(kt);
  kt = kt*rcs2; /* Without normalisation kT = cs^2 */

  init_physics();

  /* Initialise the relaxation times */

  rtau_shear = 2.0 / (1.0 + 6.0*get_eta_shear());
  rtau_bulk  = 2.0 / (1.0 + 6.0*get_eta_bulk());

  tau_s = 1.0/rtau_shear;
  tau_b = 1.0/rtau_bulk;

  /* Initialise the stress variances */

  var_bulk =
    sqrt(kt)*sqrt(2.0/9.0)*sqrt((tau_b + tau_b - 1.0)/(tau_b*tau_b));
  var_shear =
    sqrt(kt)*sqrt(1.0/9.0)*sqrt((tau_s + tau_s - 1.0)/(tau_s*tau_s));

  /* Collision global force on fluid defaults to zero. */

  siteforce[X] = 0.0;
  siteforce[Y] = 0.0;
  siteforce[Z] = 0.0;

  p = RUN_get_double_parameter_vector("force", siteforce);

  /* Information */

  info("\nModel physics:\n");
  info("Shear viscosity: %f\n", get_eta_shear());
  info("Relaxation time: %f\n", tau_s);
  info("Bulk viscosity : %f\n", get_eta_bulk());
  info("Relaxation time: %f\n", tau_b);
  info("Isothermal kT:   %f\n", get_kT());


  /* Ghost modes */

  p = RUN_get_string_parameter("ghost_modes", tmp, 128);
  if (strcmp(tmp, "off") == 0) nmodes_ = NHYDRO;

  info("\nGhost modes\n");
  if (nmodes_ == NHYDRO) {
    info("[User   ] Ghost modes have been switched off.\n");
  }
  else {
    info("[Default] All modes (hydrodynamic and ghost) are active\n");
  }

  /* Ginzburg / d'Humieres */

  p = RUN_get_string_parameter("ginzburg-dhumieres", tmp, 128);
  if (p == 1 && strcmp(tmp, "off") == 0) p = 0;

  if (p == 0) {
    tau_g = 1.0/rtau_ghost;
    info("[Default] Ghost mode relaxation time: %f\n", tau_g);
  }
  else {
    /* The leading factor here is a fudge-factor which must be in
     * range 8-12 */
    rtau_ghost = 12.0*(2.0 - rtau_shear)/(8.0 - rtau_shear);
    tau_g = 1.0/rtau_ghost;
    info("[User   ] Ginzburg-D'Humieres relaxation time requested: %f\n",
	 tau_g);
  }

  /* Noise variances */

  for (p = NHYDRO; p < NVEL; p++) {
    noise_var[p] = sqrt(kt/norm_[p])*sqrt((tau_g + tau_g - 1.0)/(tau_g*tau_g));
  }

  info("\n");

  return;
}


/*****************************************************************************
 *
 *  get_fluctuations_stress
 *
 *  Compute the random stress maxtrix with appropriate variances.
 *  This should be called once per active lattice site to set
 *  shat[][], which goes into the reprojection of the distributions.
 *
 *  Isothermal fluctuations following Adhikari et al., Europhys. Lett
 *  (2005).
 *
 *****************************************************************************/

void get_fluctuations_stress(double shat[3][3]) {

  double tr;
  const double r3 = (1.0/3.0);

  /* Set symetric random stress matrix (elements with unit variance) */

  shat[X][X] = ran_parallel_gaussian();
  shat[X][Y] = ran_parallel_gaussian();
  shat[X][Z] = ran_parallel_gaussian();

  shat[Y][X] = shat[X][Y];
  shat[Y][Y] = ran_parallel_gaussian();
  shat[Y][Z] = ran_parallel_gaussian();

  shat[Z][X] = shat[X][Z];
  shat[Z][Y] = shat[Y][Z];
  shat[Z][Z] = ran_parallel_gaussian();

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

  return;
}
