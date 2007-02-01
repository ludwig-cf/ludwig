/*****************************************************************************
 *
 *  collision.c
 *
 *  Collision stage routines and associated data.
 *
 *  $Id: collision.c,v 1.4 2007-02-01 17:31:24 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <string.h> /* Remove strcmp */
#include <stdlib.h> /* Remove calloc */

#include "pe.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
#include "physics.h"
#include "runtime.h"
#include "control.h"
#include "free_energy.h"
#include "lattice.h"

#include "utilities.h"
#include "communicate.h"
#include "leesedwards.h"
#include "model.h"
#include "collision.h"

extern Site * site;
extern double * phi_site;
extern struct vector * fl_force;
extern struct vector * fl_u;

/* Variables (not static) */

FVector * grad_phi;
double  * delsq_phi;
double  * rho_site;
char    * site_map;
double    siteforce[3];

/* Variables concerned with the collision */

static int    nmodes_ = NVEL;  /* Modes to use in collsion stage */

static double  mobility;       /* Order parameter Mobility   */
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

  void COLL_compute_phi_gradients(void);
  int  get_N_colloid(void);
  int  boundaries_present(void);

  /* This is the binary LB collision. First, compute order parameter
   * gradients, then Swift etal. collision stage. */

  TIMER_start(TIMER_PHI_GRADIENTS);

  MODEL_calc_phi();

  if (get_N_colloid() > 0 || boundaries_present()) {
    /* Must get gradients right so use this */ 
    COLL_compute_phi_gradients();
  }
  else {
    /* No solid objects (including cases with LE planes) use this */
    MODEL_get_gradients();
  }
  TIMER_stop(TIMER_PHI_GRADIENTS);

  MODEL_collide_binary_lb();

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
  int       xfac, yfac;

  double    mode[NVEL];              /* Modes; hydrodynamic + ghost */
  double    rho, rrho;               /* Density, reciprocal density */
  double    u[3];                    /* Velocity */
  double    s[3][3];                 /* Stress */
  double    shat[3][3];              /* random stress */

  double    force[3];                /* External force */
  double    tr_s, tr_seq;

  double * force_local;
  double * u_field;
  const double   r3     = (1.0/3.0);

  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);
  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	index = ic*xfac + jc*yfac + kc;

	if (site_map[index] != FLUID) continue;

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

	rho_site[index] = rho;

	/* Compute the local velocity, taking account of any body force */

	rrho = 1.0/rho;
	force_local = fl_force[index].c;
	u_field = fl_u[index].c;

	for (i = 0; i < 3; i++) {
	  force[i] = (siteforce[i] + force_local[i]);
	  u[i] = rrho*(u[i] + 0.5*force[i]);  
	  u_field[i] = u[i];
	}

	rho_site[index] = rho;

	/* Relax stress with different shear and bulk viscosity */

	tr_s   = 0.0;
	tr_seq = 0.0;

	for (i = 0; i < 3; i++) {
	  /* Compute trace */
	  tr_s   += s[i][i];
	  tr_seq += (rho*u[i]*u[i]);
	}

	/* Form traceless parts */
	for (i = 0; i < 3; i++) {
	  s[i][i]   -= r3*tr_s;
	}

	/* Relax each mode */
	tr_s = tr_s - rtau_bulk*(tr_s - tr_seq);

	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    s[i][j] -= rtau_shear*(s[i][j] - rho*u[i]*u[j]);
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
 *   from the distributions g.
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
  int       xfac, yfac;

  double    mode[NVEL];              /* Modes; hydrodynamic + ghost */
  double    rho, rrho;               /* Density, reciprocal density */
  double    u[3];                    /* Velocity */
  double    s[3][3];                 /* Stress */
  double    shat[3][3];              /* random stress */

  double    force[3];                /* External force */
  double    tr_s, tr_seq;

  double * force_local;
  double * u_field;
  const double   r3     = (1.0/3.0);


  double    phi, jdotc, sphidotq;    /* modes */
  double    jphi[3];
  double    sth[3][3], sphi[3][3];
  double    phi_x, phi_y, phi_z;     /* \nabla\phi */
  double    grad_phi_sq;
  double    mu;                      /* Chemical potential */
  double    s1;                      /* Diagonal part thermodynamic stress */
  double    A, B, kappa;             /* Free energy parameters */
  double    rtau2;
  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */


  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);

  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  rtau2 = 2.0 / (1.0 + 6.0*mobility);

  /* Free energy parameters */
  A = free_energy_A();
  B = free_energy_B();
  kappa = free_energy_K();

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	index = ic*xfac + jc*yfac + kc;

	if (site_map[index] != FLUID) continue;

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

	rho_site[index] = rho;

	/* Compute the local velocity, taking account of any body force */

	rrho = 1.0/rho;
	force_local = fl_force[index].c;
	u_field = fl_u[index].c;

	for (i = 0; i < 3; i++) {
	  force[i] = (siteforce[i] + force_local[i]);
	  u[i] = rrho*(u[i] + 0.5*force[i]);  
	  u_field[i] = u[i];
	}

	rho_site[index] = rho;


	/* Compute the thermodynamic component of the stress */

	/* Chemical potential */
	phi = phi_site[index];
	mu = phi*(A + B*phi*phi) - kappa*delsq_phi[index];

	phi_x = (grad_phi + index)->x;
	phi_y = (grad_phi + index)->y;
	phi_z = (grad_phi + index)->z;

	grad_phi_sq = phi_x*phi_x + phi_y*phi_y + phi_z*phi_z;
	s1 = 0.5*phi*phi*(A + (3.0/2.0)*B*phi*phi)
	  - kappa*(phi*delsq_phi[index] + 0.5*grad_phi_sq);

	sth[0][0] = s1 + kappa*phi_x*phi_x;
	sth[0][1] =      kappa*phi_x*phi_y;
	sth[0][2] =      kappa*phi_x*phi_z;
	sth[1][0] =      sth[0][1];
	sth[1][1] = s1 + kappa*phi_y*phi_y;
	sth[1][2] =      kappa*phi_y*phi_z;
	sth[2][0] =      sth[0][2];
	sth[2][1] =      sth[1][2];
	sth[2][2] = s1 + kappa*phi_z*phi_z;

	/* Relax stress with different shear and bulk viscosity */

	tr_s   = 0.0;
	tr_seq = 0.0;

	for (i = 0; i < 3; i++) {
	  /* Compute trace */
	  tr_s   += s[i][i];
	  tr_seq += (rho*u[i]*u[i] + sth[i][i]);
	}

	/* Form traceless parts */
	for (i = 0; i < 3; i++) {
	  s[i][i]   -= r3*tr_s;
	  sth[i][i] -= r3*tr_seq;
	}

	/* Relax each mode */
	tr_s = tr_s - rtau_bulk*(tr_s - tr_seq);

	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    s[i][j] -= rtau_shear*(s[i][j] - rho*u[i]*u[j] - sth[i][j]);
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

	/* Relax order parameters modes. */

	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    sphi[i][j] = phi*u[i]*u[j] + mobility*mu*d_[i][j];
	  }
	  jphi[i] = phi*u[i];
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
 *- \c Buffers:   sets .halo, .rho, .phi, .grad_phi
 *- \c Version:   2.0b1
 *- \c Last \c updated: 01/03/2002 by JCD
 *- \c Authors:   P. Bladon and JC Desplat
 *- \c Note:      none
 */
/*----------------------------------------------------------------------------*/

void MODEL_init( void ) {

  int     i,j,k,ind,xfac,yfac,N_sites;
  int     N[3];
  int     offset[3];
  double   phi;
  double   phi0, rho0;

  rho0 = get_rho0();
  phi0 = get_phi0();

  get_N_local(N);
  get_N_offset(offset);

  xfac = (N[Y]+2)*(N[Z]+2);
  yfac = (N[Z]+2);

  N_sites = (N[X]+2)*(N[Y]+2)*(N[Z]+2);

  /* First allocate memory for site map */
  if((site_map = (char*)calloc(N_sites,sizeof(char))) == NULL)
    {
      fatal("MODEL_init(): failed to allocate %d bytes for site_map[]\n",
	    N_sites*sizeof(char));
    }  
  
  /* Now setup the rest of the simulation */

  /* Allocate memory */

  info("Requesting %d bytes for grad_phi\n", N_sites*sizeof(FVector));
  info("Requesting %d bytes for delsq_phi\n", N_sites*sizeof(double));
  info("Requesting %d bytes for rho_site\n", N_sites*sizeof(double));

  grad_phi  = (FVector*)calloc(N_sites,sizeof(FVector));
  delsq_phi = (double  *)calloc(N_sites,sizeof(double  ));
  rho_site  = (double  *)calloc(N_sites,sizeof(double  ));

  if(grad_phi==NULL || delsq_phi==NULL || rho_site==NULL)
    {
      fatal("MODEL_init(): failed to allocate %d bytes for vels\n",
	    2*N_sites*(sizeof(FVector)+sizeof(double)));
    }

  init_site();
  LATT_allocate_phi(N_sites);
  LATT_allocate_force(N_sites);
  latt_allocate_velocity(N_sites);

  
  /*
   * A number of options are offered to start a simulation:
   * 1. Read distribution functions site from file, then simply calculate
   *    all other properties (rho/phi/gradients/velocities)
   * 6. set rho/phi/velocity to default values, automatically set etc.
   */

  RUN_get_double_parameter("noise", &noise0);

  /* Option 1: read distribution functions from file */
  if( strcmp(get_input_config_filename(0),"EMPTY") != 0 ) {

    info("Re-starting simulation at step %d with data read from "
	 "config\nfile(s) %s\n", get_step(), get_input_config_filename(0));

    /* Read distribution functions - sets both */
    COM_read_site(get_input_config_filename(0),MODEL_read_site);
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
		ind = (i-offset[X])*xfac + (j-offset[Y])*yfac + (k-offset[Z]);
		site_map[ind] = FLUID;
#ifdef _SINGLE_FLUID_
		set_rho(rho0, ind);
		set_phi(phi0, ind);
#else
		set_rho(rho0, ind); 
		set_phi(phi,  ind);
#endif
	      }
	  }
  }

}

/*****************************************************************************
 *
 *  MODEL_calc_phi
 *
 *  Recompute the value of the order parameter at all the current
 *  fluid sites (domain proper).
 *
 *  The halo regions are immediately updated to reflect the new
 *  values.
 *
 *****************************************************************************/

void MODEL_calc_phi() {

  int     i, j , k, index, p;
  int     xfac, yfac;
  int     N[3];
  double  * g;

  get_N_local(N);

  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	index = i*xfac + j*yfac + k;

	if (site_map[index] != FLUID) {
	  /* This is an undefined value... */
	  phi_site[index] = -1000.0;
	}
	else {

	  g = site[index].g;
	  phi_site[index] = g[0];

	  for (p = 1; p < NVEL; p++)
	    phi_site[index] += g[p];

	}
      }
    }
  }

  COM_halo_phi();

  return;
}

/****************************************************************************
 *
 *  RAND_init_fluctuations
 *
 *  Set variances for fluctuating lattice Boltzmann.
 *  Issues
 *    IMPORTANT: Note that the noise generation is not decomposition-
 *               independent when run in parallel.
 *
 ****************************************************************************/

void RAND_init_fluctuations() {

  int  p;
  char tmp[128];
  double tau_s, tau_b, tau_g, kt;

  p = RUN_get_double_parameter("temperature", &kt);
  kt = kt*rcs2; /* Without normalisation kT = cs^2 */
  set_kT(kt);

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
  info("Isothermal kT:   %f\n", get_kT()/rcs2);

  /* Order parameter mobility (probably to move) */

  p = RUN_get_double_parameter("mobility", &mobility);
  info("\nOrder parameter mobility M: %f\n", mobility);


  /* Ghost modes */

  p = RUN_get_string_parameter("ghost_modes", tmp);
  if (strcmp(tmp, "off") == 0) nmodes_ = NHYDRO;

  info("\nGhost modes\n");
  if (p == 1) {
    info("[User   ] Ghost modes have been switched off.\n");
  }
  else {
    info("[Default] All modes (hydrodynamic and ghost) are active\n");
  }

  /* Ginzburg / d'Humieres */

  p = RUN_get_string_parameter("ginzburg-dhumieres", tmp);
  if (p == 1 && strcmp(tmp, "off") == 0) p = 0;

  if (p == 0) {
    tau_g = 1.0/rtau_ghost;
    info("[Default] Ghost mode relaxation time: %f\n", tau_g);
  }
  else {
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

/*****************************************************************************
 *
 *  MISC_set_mean_phi
 *
 *  Compute the current mean phi in the system and remove the excess
 *  so that the mean phi is phi_global (allowing for presence of any
 *  particles or, for that matter, other solids).
 *
 *  The value of phi_global is generally (but not necessilarily) zero.
 *
 *****************************************************************************/

void MISC_set_mean_phi(double phi_global) {

  int     index, i, j, k, p;
  int     xfac, yfac;
  int     nfluid = 0;
  int     N[3];

  double  phi;
  double  phibar =  0.0;

  get_N_local(N);
  yfac = (N[Z] + 2);
  xfac = (N[Y] + 2)*yfac;

  /* Compute the mean phi in the domain proper */

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	index = xfac*i + yfac*j + k;

	if (site_map[index] != FLUID) continue;

	phi = 0.0;

	for (p = 0; p < NVEL; p++) {
	  phi += site[index].g[p];
	}

	phibar += phi;
	nfluid += 1;
      }
    }
  }

#ifdef _MPI_
  {
    int    n_total;
    double phi_total;

    /* All processes need the total phi, and number of fluid sites
     * to compute the mean */

    MPI_Allreduce(&phibar, &phi_total, 1, MPI_DOUBLE, MPI_SUM, cart_comm());
    MPI_Allreduce(&nfluid, &n_total,   1, MPI_INT,    MPI_SUM, cart_comm());

    phibar = phi_total;
    nfluid = n_total;
  }
#endif

  /* The correction requied at each fluid site is then ... */
  phi = phi_global - phibar / (double) nfluid;

  /* The correction is added to the rest distribution g[0],
   * which should be good approximation to where it should
   * all end up if there were a full reprojection. */

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	index = xfac*i + yfac*j + k;
	if (site_map[index] == FLUID) site[index].g[0] += phi;
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  latt_zero_force
 *
 *  Set the force on the lattice sites to zero.
 *
 *****************************************************************************/

void latt_zero_force() {

  int ic, jc, kc, index, xfac, yfac;
  int N[3];

  get_N_local(N);
  yfac = (N[Z] + 2);
  xfac = (N[Y] + 2)*yfac;

  /* Compute the mean phi in the domain proper */

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	index = xfac*ic + yfac*jc + kc;

	fl_force[index].c[X] = 0.0;
	fl_force[index].c[Y] = 0.0;
	fl_force[index].c[Z] = 0.0;	
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  MISC_fluid_volume
 *
 *  What is the current fluid volume? This is useful when one has a
 *  gravitational force on moving particles and there is then a need
 *  to compute an equal and opposite force in the fluid.
 *
 *  The value is computed as a double.
 *
 *****************************************************************************/

double MISC_fluid_volume() {

  double  v = 0.0;

  int     index, i, j, k;
  int     xfac, yfac;
  int     N[3];

  get_N_local(N);
  yfac = (N[Z] + 2);
  xfac = (N[Y] + 2)*yfac;

  /* Look for fluid nodes (not halo) */

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	index = xfac*i + yfac*j + k;
	if (site_map[index] == FLUID) v += 1.0;
      }
    }
  }

#ifdef _MPI_
  {
    double v_total;

    /* All processes need the total */

    MPI_Allreduce(&v, &v_total, 1, MPI_DOUBLE, MPI_SUM, cart_comm());
    v = v_total;
  }
#endif

  return v;
}

/******************************************************************************
 *
 *  MISC_curvature
 *
 *  This function looks at the phi field and computes the curvature
 *  maxtrix. This can then be used to estimate the domain lengths
 *  in the coordinate directions and in the 'natural' directions.
 *
 *  The natural lengths are just reported in decreasing order.
 *
 *  A version of the above which also prints out the elements
 *  of the curvature matrix along with the length estimates.
 *
 *  See Paul's notes for further details.
 *
 ****************************************************************************/

void  MISC_curvature() {

#ifdef _SINGLE_FLUID_
  /* Do nothing */
#else

  double eva1, eva2, eva3;
  double alpha, beta; 
  double lx, ly, lz;
  double L1, L2, L3;

  int i, j, k, index;
  int xfac, yfac;
  int N[3];

  FVector dphi;

  double eve1[3], eve2[3], eve3[3];
  double sum[6];
  double abnorm, rv;

  void eigen3(double, double, double, double, double, double,
	      double * , double * , double *, double *, double *, double *); 

  get_N_local(N);
  xfac = (N[Y]+2)*(N[Z]+2);
  yfac = (N[Z]+2);

  for (i = 0; i < 6; i++) {
    sum[i] = 0.0;
  }

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1;k <= N[Z]; k++) {

	index = i*xfac + j*yfac + k;
            
	dphi = grad_phi[index];
	sum[0] += dphi.x*dphi.x;
	sum[1] += dphi.x*dphi.y;
	sum[2] += dphi.x*dphi.z;
	sum[3] += dphi.y*dphi.y;
	sum[4] += dphi.y*dphi.z;
	sum[5] += dphi.z*dphi.z;
      }
    }
  }

#ifdef _MPI_
  /* Note that we use Reduce here, so only process 0 in
   * MPI_COMM_WORLD gets the correct total. This is approriate
   * as info() is used to give the results. */
 {
   double gsum[6];

   for (i = 0; i < 6; i++) {
     gsum[i] = sum[i];
   }

   MPI_Reduce(gsum, sum, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 }

#endif

  rv = 1.0 / (L(X)*L(Y)*L(Z));
  sum[0] *= rv;
  sum[1] *= rv;
  sum[2] *= rv;
  sum[3] *= rv;
  sum[4] *= rv;
  sum[5] *= rv;
 
  /* This is the phi^4 free energy with A = B */
  abnorm = 4.0/(3.0*interfacial_width());
  lx = abnorm / sum[0];
  ly = abnorm / sum[3];
  lz = abnorm / sum[5];
    

  eigen3(sum[0],sum[1],sum[2],sum[3],sum[4],sum[5],
	 &eva1,&(eve1[0]),&eva2,&(eve2[0]),&eva3,&(eve3[0]));


  /* Sort the eva values in ascending order */
  if( eva1 < eva2 ){
    rv = eva2; eva2 = eva1; eva1 = rv;
    rv=eve2[0]; eve2[0]=eve1[0]; eve1[0]=rv;
    rv=eve2[1]; eve2[1]=eve1[1]; eve1[1]=rv;
    rv=eve2[2]; eve2[2]=eve1[2]; eve1[2]=rv;
  }
  if( eva1 < eva3 ){
    rv = eva3; eva3 = eva1; eva1=rv;
    rv=eve3[0]; eve3[0]=eve1[0]; eve1[0]=rv;
    rv=eve3[1]; eve3[1]=eve1[1]; eve1[1]=rv;
    rv=eve3[2]; eve3[2]=eve1[2]; eve1[2]=rv;
  }
  if( eva2 < eva3 ){
    rv = eva3; eva3 = eva2; eva2 = rv;
    rv=eve3[0]; eve3[0]=eve2[0]; eve2[0]=rv;
    rv=eve3[1]; eve3[1]=eve2[1]; eve2[1]=rv;
    rv=eve3[2]; eve3[2]=eve2[2]; eve2[2]=rv;
  }
  
  /* Check to see if any of the eva values are zero. If so, set associated
     length scale to zero. */

  if( eva1 < 1e-10 ){
    L1 = 0.0;
  }
  else{
    L1 = abnorm / eva1;
  }
  if( eva2 < 1e-10 ){
    L2 = 0.0;
  }
  else{
    L2 = abnorm / eva2;
  }
  if( eva3 < 1e-10 ){
    L3 = 0.0;
  }
  else{
    L3 = abnorm / eva3;
  }

  /* Calculate the angles. */
  if( fabs(eve1[1]) < 1e-10){
    alpha = 0.5*PI;
    beta  = 0.5*PI;
  }
  else{
    alpha = atan(eve1[0]/eve1[1]);
    beta  = atan(eve1[2]/eve1[1]);
  }

  info("\nCurvature statistics [ t, lx, ly, lz, L1, L2, L3, alpha, beta]\n");
  info("%d  %7g %7g %7g %7g %7g %7g %7g %7g\n", get_step(),
       lx, ly, lz, L1, L2, L3, alpha, beta);
#endif

  return;
}
