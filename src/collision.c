
#include <stdio.h>
#include <math.h>
#include <string.h> /* Remove strcmp */
#include <stdlib.h> /* Remove calloc */

#include "pe.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
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
extern double q_[NVEL][3][3];
extern struct vector * fl_force;

/* Variables (not static) */

FVector * grad_phi;
double  * delsq_phi;
double  * rho_site;
char    * site_map;
double    siteforce[3];

/* Variables concerned with the collision */
static double  mobility;       /* Order parameter Mobility   */
static double  noise0 = 0.1;   /* Initial noise amplitude    */
static double rtau_shear;      /* Inverse relaxation time for shear modes */
static double rtau_bulk;       /* Inverse relaxation time for bulk modes */
static double var_shear;       /* Variance for shear mode fluctuations */
static double var_bulk;        /* Variance for bulk mode fluctuations */


#ifdef _SINGLE_FLUID_

/*****************************************************************************
 *
 *  MODEL_collide_multirelaxation
 *
 *  Collision with potentially different relaxation for different modes.
 *  
 *  This routine is currently model independent, except that
 *  it is assumed that p = 0 is the null vector in the set.
 *
 *****************************************************************************/

void MODEL_collide_multirelaxation() {

  int      N[3];
  int      ic, jc, kc, index;       /* site indices */
  int      p;                       /* velocity index */
  int      i, j;                    /* summed over indices ("alphabeta") */
  int      xfac, yfac;

  double    u[3];                    /* Velocity */
  double    s[3][3];                 /* Stress */
  double    shat[3][3];              /* random stress */
  double    rho, rrho;               /* Density, reciprocal density */
  double *  f;

  double    udotc;
  double    sdotq;

  double    force[3];                /* External force */
  double    tr_s, tr_seq;

  const double   r2rcs4 = 4.5;      /* The constant 1 / 2 c_s^4 */
  const double   r3     = (1.0/3.0);

  double fghost[NVEL];              /* Model-dependent ghost modes */

  double * force_local;

  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);
  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  for (p = 0; p < NVEL; p++) {
    fghost[p] = 0.0;
  }

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	index = ic*xfac + jc*yfac + kc;

	if (site_map[index] != FLUID) continue;

	f = site[index].f;

	rho  = f[0];
	u[0] = 0.0;
	u[1] = 0.0;
	u[2] = 0.0;

	for (p = 1; p < NVEL; p++) {
	  rho  += f[p];
	  u[0] += f[p]*cv[p][0];
	  u[1] += f[p]*cv[p][1];
	  u[2] += f[p]*cv[p][2];
	}

	rho_site[index] = rho;

	/* Compute the local velocity, taking account of any body force */

	rrho = 1.0/rho;
	force_local = fl_force[index].c;

	for (i = 0; i < 3; i++) {
	  force[i] = 0.5*(siteforce[i] + force_local[i]);
	  u[i] = rrho*(u[i] + force[i]);  
	}

	/* Relax stress with different shear and bulk viscosity */

	tr_s   = 0.0;
	tr_seq = 0.0;

	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    /* Compute s */
	    s[i][j] = 0.0;
	    shat[i][j] = 0.0;

	    for (p = 0; p < NVEL; p++) {
	      s[i][j] += f[p]*q_[p][i][j];
	    }
	  }
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
	  }
	}

	/* Now update the distribution */

#ifdef _NOISE_
	get_fluctuations_stress(shat);
	get_ghosts(fghost);
#endif

	for (p = 0; p < NVEL; p++) {

	  udotc = 0.0;
	  sdotq = 0.0;

	  for (i = 0; i < 3; i++) {
	    udotc += (u[i] + rrho*force[i])*cv[p][i];
	    for (j = 0; j < 3; j++) {
	      sdotq += (s[i][j] + shat[i][j])*q_[p][i][j];
	    }
	  }

	  /* Reproject */
	  f[p] = wv[p]*(rho + rho*udotc*rcs2 + sdotq*r2rcs4 + fghost[p]);

	  /* Next p */
	}

	/* Next site */
      }
    }
  }
 
 TIMER_stop(TIMER_COLLIDE);

  return;
}


#else

/*****************************************************************************
 *
 *  MODEL_collide_multirelaxation
 *
 *  Collision with different relaxation for different modes.
 *
 *  Binary fluid version. This is currently being tested, but
 *  we think it's correct.
 *
 *  At each site, compute the denisty, order parameter, velocity,
 *  order parameter flux, and contribution to the stress tensor
 *  from the thermodynamic sector and the density. The stress is
 *  relaxed toward the equilibrium value, while the order parameter
 *  flux is also relaxed toward equilibrium.
 *
 *  The order parameter 'stress' is set to the equilibrium value
 *  following Ronojoy's suggestion. Note the order parameter
 *  mobility now only enters via the relaxation time for the
 *  order parameter.
 *
 *  Body forces assume a single relaxation time at the moment.
 *
 *****************************************************************************/

void MODEL_collide_multirelaxation() {

  int      N[3];
  int      ic, jc, kc, index;       /* site indices */
  int      p;                       /* velocity index */
  int      i, j;                    /* summed over indices ("alphabeta") */
  int      xfac, yfac;

  double    u[3];                    /* Velocity */
  double    jphi[3];                 /* Order parameter flux */
  double    s[3][3];                 /* Stress */
  double   shat[3][3];              /* Random stress */
  double    sth[3][3];               /* Equilibrium stress (thermodynamic) */
  double    sphi[3][3];              /* Order parameter "stress" */
  double    rho, rrho;               /* Density, reciprocal density */
  double    rtau2;                   /* Reciprocal \tau \tau_2 */
  double    tr_s, tr_seq;
  double *  f;
  double *  g;

  double    udotc;
  double    jdotc;
  double    sdotq, sphidotq;

  double    force[3];                /* External force */

  double    phi;                     /* Order parameter */
  double    phi_x, phi_y, phi_z;     /* \nabla\phi */
  double    grad_phi_sq;
  double    mu;                      /* Chemical potential */
  double    s1;                      /* Diagonal part of thermodynamic stress */
  double    A, B, kappa;             /* Free energy parameters */

  const double r2   = 0.5;
  const double r3   = 1.0/3.0;
  const double c3r2 = 1.5;

  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  double    fghost[NVEL];

  double    dp0;

  double * force_local;

  void COLL_compute_phi_gradients(void);
  int  get_N_colloid(void);
  int  boundaries_present(void);

  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);

  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  rtau2 = 2.0 / (1.0 + 6.0*mobility);

  /* Order parameter and order parameter gradients */
  A = free_energy_A();
  B = free_energy_B();
  kappa = free_energy_K();

  MODEL_calc_phi();

  if (get_N_colloid() > 0 || boundaries_present()) {
    /* Must get gradients right so use this */ 
    COLL_compute_phi_gradients();
  }
  else {
    /* No solid objects (including cases with LE planes) use this */
    MODEL_get_gradients();
  }

  /* Initialise the ghost part of the distribution */
  for (p = 0; p < NVEL; p++) {
    fghost[p] = 0.0;
  }

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	index = ic*xfac + jc*yfac + kc;

	if (site_map[index] != FLUID) continue;

	f = site[index].f;
	g = site[index].g;

	force_local = fl_force[index].c;

	rho  = f[0];
	phi  = g[0];
	u[0] = 0.0;
	u[1] = 0.0;
	u[2] = 0.0;
	jphi[0] = 0.0;
	jphi[1] = 0.0;
	jphi[2] = 0.0;

	for (p = 1; p < NVEL; p++) {
	  rho  += f[p];
	  phi  += g[p];
	  u[0] += f[p]*cv[p][0];
	  u[1] += f[p]*cv[p][1];
	  u[2] += f[p]*cv[p][2];
	  jphi[0] += g[p]*cv[p][0];
	  jphi[1] += g[p]*cv[p][1];
	  jphi[2] += g[p]*cv[p][2];
	}

	/* Body force is added to the velocity going into collision;
	 * the variable "force" holds 0.5 x actual force. */

	rrho = 1.0/rho;

	for (i = 0; i < 3; i++) {
	  force[i] = 0.5*(siteforce[i] + force_local[i]);
	  u[i] = rrho*(u[i] + force[i]);
	}

	/* Chemical potential */
	mu = phi*(A + B*phi*phi) - kappa*delsq_phi[index];

	/* Thermodynamic part of the stress */

	phi_x = (grad_phi + index)->x;
	phi_y = (grad_phi + index)->y;
	phi_z = (grad_phi + index)->z;

	grad_phi_sq = phi_x*phi_x + phi_y*phi_y + phi_z*phi_z;
	s1 = r2*phi*phi*(A + c3r2*B*phi*phi)
	  - kappa*(phi*delsq_phi[index] + r2*grad_phi_sq);

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
	  for (j = 0; j < 3; j++) {
	    /* Compute s */
	    s[i][j] = 0.0;
	    shat[i][j] = 0.0;

	    for (p = 0; p < NVEL; p++) {
	      s[i][j] += f[p]*q_[p][i][j];
	    }
	  }
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
	    
	    /* Order parameter stress is fixed to the equilibrium value
	     * related to the chemical potential */

	    sphi[i][j] = phi*u[i]*u[j] + mu*d_[i][j];
	  }

	  /* Order parameter flux is relaxed toward equilibrium value */

	  jphi[i] = jphi[i] - rtau2*(jphi[i] - phi*u[i]);
	}

	/* Now update the distribution */

#ifdef _NOISE_
	get_fluctuations_stress(shat);
	get_ghosts(fghost);
#endif

	for (p = 0; p < NVEL; p++) {

	  dp0 = (p == 0);

	  udotc    = 0.0;
	  jdotc    = 0.0;
	  sdotq    = 0.0;
	  sphidotq = 0.0;

	  for (i = 0; i < 3; i++) {
	    udotc += (u[i] + rrho*force[i])*cv[p][i];
	    jdotc += jphi[i]*cv[p][i];
	    for (j = 0; j < 3; j++) {
	      sdotq += (s[i][j] + shat[i][j])*q_[p][i][j];
	      sphidotq += sphi[i][j]*q_[p][i][j];
	    }
	  }

	  /* Project all this back to the distributions. The magic
	   * here is to move phi into the non-propagating distribution. */

	  f[p] = wv[p]*(rho + rho*udotc*rcs2 + sdotq*r2rcs4 + fghost[p]);
	  g[p] = wv[p]*(          jdotc*rcs2 + sphidotq*r2rcs4) + phi*dp0;

	}

	/* Next site */
      }
    }
  }


  TIMER_stop(TIMER_COLLIDE);

  return;
}

#endif /* _SINGLE_FLUID_ */

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
 *- \c See \c also: COM_init(), LE_init(), MODEL_process_options(), 
 *                COM_read_link(), MODEL_read_site_data()
 *- \c Note:      none
 */
/*----------------------------------------------------------------------------*/

void MODEL_init( void ) {

  int     i,j,k,ind,xfac,yfac,N_sites;
  int     N[3];
  int     offset[3];
  double   phi;
  double   phi0, rho0;

  void le_init_transitional(void);

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

  allocate_site(N_sites);
  LATT_allocate_phi(N_sites);
  LATT_allocate_force(N_sites);

  /* KS: should be called after MODEL_init() */
  le_init_transitional();
  
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
  /* Option 6: set rho/phi to defaults */
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

  /*
   * Initialise Lees-Edwards buffers (if required): needs to be called *before*
   * any call to MODEL_get_gradients() but can only take place *after* the
   * distribution functions have been set
   */
  LE_update_buffers(SITE_AND_PHI);

}

/*****************************************************************************
 *
 *  MODEL_get_momentum_at_site
 *
 *  Return momentum density at lattice node index.
 *
 *****************************************************************************/

FVector MODEL_get_momentum_at_site(const int index) {

  FVector mv;
  double  * f;
  int     p;

  mv.x = 0.0;
  mv.y = 0.0;
  mv.z = 0.0;

  f  = site[index].f;

  for (p = 0; p < NVEL; p++) {
    mv.x += cv[p][0]*f[p];
    mv.y += cv[p][1]*f[p];
    mv.z += cv[p][2]*f[p];
  }

  return mv;
}


/*----------------------------------------------------------------------------*/
/*!
 * Computes rho for all sites (including halos) with
 * \f[ \rho = \sum_{i=1}^{NVEL} f_i \f]
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: void
 *- \c Returns:   void
 *- \c Buffers:   uses .halo, sets .rho
 *- \c Version:   2.0b1
 *- \c Last \c updated: 27/01/2002 by JCD
 *- \c Authors:   P. Bladon and JC Desplat
 *- \c See \c also: MODEL_get_rho(), MODEL_set_rho(), MODEL_calc_phi()
 *- \c Note:      site.f in halos region must be up-to-date. COM_halo() is
 *                therefore called from within this routine
 */
/*----------------------------------------------------------------------------*/

void MODEL_calc_rho( void ) {

  int i,j,k,ind,xfac,yfac,p;
  int N[3];
  double *f;

  get_N_local(N);
  yfac = (N[Z]+2);
  xfac = (N[Z]+2)*(N[Y]+2);
  
  COM_halo();
  
  for (i = 0; i <= N[X] + 1; i++)
    for (j = 0; j <= N[Y] + 1; j++)
      for (k = 0; k <= N[Z] + 1; k++)
	{
	  ind = i*xfac + j*yfac + k;
	  
	  f = site[ind].f;

	  rho_site[ind] = 0.0;
	  for (p = 0; p < NVEL; p++) {
	    rho_site[ind] += f[p];
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
  double tau_s, tau_b;

  model_init();

  /* Initialise the relaxation times */

  rtau_shear = 2.0 / (1.0 + 6.0*get_eta_shear());
  rtau_bulk  = 2.0 / (1.0 + 6.0*get_eta_bulk());

  tau_s = 1.0/rtau_shear;
  tau_b = 1.0/rtau_bulk;

  /* Initialise the stress variances */

  var_bulk =
    sqrt(get_kT())*sqrt(2.0/9.0)*sqrt((tau_b + tau_b - 1.0)/(tau_b*tau_b));
  var_shear =
    sqrt(get_kT())*sqrt(1.0/9.0)*sqrt((tau_s + tau_s - 1.0)/(tau_s*tau_s));

  init_ghosts(get_kT());

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

  /* Order parameter mobility (probably to move) */

  p = RUN_get_double_parameter("mobility", &mobility);
  info("\nOrder parameter mobility M: %f\n", mobility);

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
