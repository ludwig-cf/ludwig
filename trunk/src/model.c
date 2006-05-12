#include "globals.h"

#include "pe.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
#include "runtime.h"
#include "control.h"
#include "cartesian.h"
#include "free_energy.h"
#include "lattice.h"

static void    MODEL_set_rho(const double, const int);
static void    MODEL_set_phi(const double, const int);

static void    MODEL_read_site_asc( FILE * );
static void    MODEL_read_site_bin( FILE * );
static void    MODEL_write_site_asc( FILE *, int, int );
static void    MODEL_write_velocity_asc( FILE *, int, int );
static void    MODEL_write_rho_asc( FILE *, int, int );
static void    MODEL_write_phi_asc( FILE *, int, int );
static void    MODEL_write_rho_phi_asc( FILE *, int, int );
static void    MODEL_write_site_bin( FILE *, int, int );
static void    MODEL_write_velocity_bin( FILE *, int, int );
static void    MODEL_write_rho_bin( FILE *, int, int );
static void    MODEL_write_phi_bin( FILE *, int, int );
static void    MODEL_write_rho_phi_bin( FILE *, int, int );

/* Variables (not static) */

FVector * grad_phi;
Float   * delsq_phi;
Float   * rho_site;

Global    gbl;
char    * site_map;
double    siteforce[3];

extern double * phi_site;


/* Generic functions for output */

void (*MODEL_write_site)( FILE *, int, int );
void (*MODEL_write_phi)( FILE *, int, int );

static void (*MODEL_write_velocity)( FILE *, int, int );
static void (*MODEL_write_rho)( FILE *, int, int );
static void (*MODEL_write_rho_phi)( FILE *, int, int );

/* Generic functions for input */
static void (*MODEL_read_site)( FILE * );



/* Variables concerned with the collision */

static double eta_shear;       /* Shear viscosity */
static double eta_bulk;        /* Bulk viscosity */
static double rtau_shear;      /* Inverse relaxation time for shear modes */
static double rtau_bulk;       /* Inverse relaxation time for bulk modes */
static double var_shear;       /* Variance for shear mode fluctuations */
static double var_bulk;        /* Variance for bulk mode fluctuations */
       double _q[NVEL][3][3];  /* Q matrix */



#ifdef _SINGLE_FLUID_

/*****************************************************************************
 *
 *  MODEL_collide_multirelaxation
 *
 *  Collision with potentially different relaxation for different modes.
 *  From Ronojoy.
 *
 *  
 *  This routine is currently model independent, except that
 *  it is assumed that p = 0 is the null vector in the set.
 *
 *  The BGK form replaces the multirelaxation as :
 *    usq  = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
 *    feq  = wv[p]*rho*(1.0 + rcs2*udotc + r2rcs4*udotc*udotc - r2rcs2*usq);
 *    f[p] = f[p] - rtau*(f[p] - feq);
 *  with appropriate noise variances if required.
 *
 *****************************************************************************/

void MODEL_collide_multirelaxation() {

  int      N[3];
  int      ic, jc, kc, index;       /* site indices */
  int      p;                       /* velocity index */
  int      i, j;                    /* summed over indices ("alphabeta") */
  int      xfac, yfac;

  Float    u[3];                    /* Velocity */
  Float    s[3][3];                 /* Stress */
  double   shat[3][3];              /* random stress */
  Float    rho, rrho;               /* Density, reciprocal density */
  Float    rtau;                    /* Reciprocal \tau */
  Float *  f;

  Float    udotc;
  Float    sdotq;
  Float    cdotf;

  Float    force[3];                /* External force */

  const Float    rcs2   = 3.0;      /* The constant 1 / c_s^2 */
  const Float    r2rcs4 = 4.5;      /* The constant 1 / 2 c_s^4 */

  double fghost[NVEL];              /* Model-dependent ghost modes */

  extern FVector * _force;
  extern double siteforce[3];

  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);
  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  rtau = 2.0 / (1.0 + 6.0*get_eta_shear());

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

	rrho = 1.0/rho;
	u[0] *= rrho;
	u[1] *= rrho;
	u[2] *= rrho;

	/* The local body force. */
	/* Note the "global" force (gravity) is still constant. */ 

	force[0] = 0.5*(siteforce[X] + (_force + index)->x);
	force[1] = 0.5*(siteforce[Y] + (_force + index)->y);
	force[2] = 0.5*(siteforce[Z] + (_force + index)->z);

	/* Compute the velocity, taking account of any body force */

	for (i = 0; i < 3; i++) {
	  u[i] += rrho*force[i];  
	}

	/* Stress */

	for (i = 0; i < 3; i++) {
	  for (j = 0; j < 3; j++) {
	    s[i][j] = 0.0;
	    shat[i][j] = 0.0;

	    for (p = 0; p < NVEL; p++) {
	      s[i][j] += f[p]*_q[p][i][j];
	    }

	    /* Relax this stress mode */
	    s[i][j] = s[i][j] - rtau*(s[i][j] - rho*u[i]*u[j]);

	    /* Add body force terms to give post-collision stress */
	    s[i][j] = s[i][j] + (2.0-rtau)*(u[i]*force[j] + force[i]*u[j]);
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
	      sdotq += (s[i][j] + shat[i][j])*_q[p][i][j];
	    }
	  }
 
#ifdef _BGK_
	  /* b acts as u^2 and c as the equilibrium feq */
	  b = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
	  c = wv[p]*rho*(1.0 + rcs2*udotc + r2rcs4*udotc*udotc - 0.5*rcs2*b);
	  f[p] = f[p] - rtau*(f[p] - c);

	  /* Body force (Guo et al.) */

	  b = 0.0;
	  c = 0.0;
	  cdotf = 0.0;

	  for (i = 0; i < 3; i++) {
	    b     += (cv[p][i] - u[i])*fxyz[i];
	    c     += cv[p][i]*u[i];
	    cdotf += cv[p][i]*fxyz[i];
	  }

	  f[p] = f[p] + 0.5*(2.0-rtau)*wv[p]*rcs2*(b + rcs2*c*cdotf);
#else
	  /* Reproject */
	  f[p] = wv[p]*(rho + rho*udotc*rcs2 + sdotq*r2rcs4 + fghost[p]);

#endif

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

  Float    u[3];                    /* Velocity */
  Float    jphi[3];                 /* Order parameter flux */
  Float    s[3][3];                 /* Stress */
  double   shat[3][3];              /* Random stress */
  Float    sth[3][3];               /* Equilibrium stress (thermodynamic) */
  Float    sphi[3][3];              /* Order parameter "stress" */
  Float    rho, rrho;               /* Density, reciprocal density */
  Float    rtau2;                   /* Reciprocal \tau \tau_2 */
  Float    tr_s, tr_seq;
  Float *  f;
  Float *  g;

  Float    udotc;
  Float    jdotc;
  Float    sdotq, sphidotq;
  Float    dij;

  Float    force[3];                /* External force */
  Float    cdotf;

  Float    phi;                     /* Order parameter */
  Float    phi_x, phi_y, phi_z;     /* \nabla\phi */
  Float    grad_phi_sq;
  Float    mu;                      /* Chemical potential */
  Float    s1;                      /* Diagonal part of thermodynamic stress */
  Float    A, B, kappa;             /* Free energy parameters */

  Float    phi0;

  const Float r2   = 0.5;
  const Float r3   = 1.0/3.0;
  const Float c3r2 = 1.5;

  const Float rcs2   = 3.0;         /* The constant 1 / c_s^2 */
  const Float rcs4   = 9.0;         /* The constant 1 / c_s^4 */
  const Float r2rcs2 = 1.5;         /* The constant 1 / 2 c_s^2 */
  const Float r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  double    fghost[NVEL];

  Float    dp0;

  extern FVector * _force;

  TIMER_start(TIMER_COLLIDE);

  get_N_local(N);

  yfac = (N[Z]+2);
  xfac = (N[Y]+2)*yfac;

  rtau2 = 2.0 / (1.0 + 6.0*gbl.mobility);

  A = free_energy_A();
  B = free_energy_B();
  kappa = free_energy_K();

  MODEL_calc_phi();

#ifdef _COLLOIDS_
  /* Allows colloids */
  COLL_compute_phi_gradients();
#else
  /* Allows LE planes */
  MODEL_get_gradients();
#endif
  /* (And the twain cannot meet!) */

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

	rrho = 1.0/rho;
	u[0] *= rrho;
	u[1] *= rrho;
	u[2] *= rrho;

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

	/* Body force is added to the velocity going into collision;
	 * the variable "force" holds 0.5 x actual force. */

	force[0] = 0.5*(siteforce[X] + (_force + index)->x);
	force[1] = 0.5*(siteforce[Y] + (_force + index)->y);
	force[2] = 0.5*(siteforce[Z] + (_force + index)->z);

	for (i = 0; i < 3; i++) {
	  u[i] += rrho*force[i];
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
	      s[i][j] += f[p]*_q[p][i][j];
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
	    dij = (i == j);
	    s[i][j] -= rtau_shear*(s[i][j] - rho*u[i]*u[j] - sth[i][j]);
	    s[i][j] += dij*r3*tr_s;

	    /* Correction from body force (assumes equal relaxation times) */

	    s[i][j] += (2.0-rtau_shear)*(u[i]*force[j] + force[i]*u[j]);
	    
	    /* Order parameter stress is fixed to the equilibrium value
	     * related to the chemical potential */

	    sphi[i][j] = phi*u[i]*u[j] + mu*dij;
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
	      sdotq += (s[i][j] + shat[i][j])*_q[p][i][j];
	      sphidotq += sphi[i][j]*_q[p][i][j];
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
 * Sets the velocity of the site s to a given value u via the distribution
 * functions
 *
 *- \c Options:   _TRACE_
 *- \c Arguments:
 *  -# \c FVector \c u: value of velocity the site will be set to
 *  -# \c Site \c *s: target site
 *- \c Returns:   void
 *- \c Buffers:   uses .grad_phi, invalidates .halo, .rho?, .phi?, .grad_phi?,
 *                .vels
 *- \c Version:   2.0
 *- \c Last \c updated: 12/01/2003 by JCD
 *- \c Authors:   P. Bladon and JC Desplat
 *- \c See \c also: MODEL_calc_vels(), MODEL_get_velocity(), MODEL_set_rho(),
 *                MODEL_set_phi()
 *- \c Note:      only performs atomic update for site s. If this function is
 *                called to set the velocity of a halo site, it is the user's
 *                responsibility to ensure that the halo is up-to-date
 */
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*!
 * Careful: "atrest" indicates steady state, not equilibrium 
 */
/*----------------------------------------------------------------------------*/

void MODEL_set_distributions_atrest( FVector u, FTensor grad_u, Site *ps )
{
  int      N[3];
  int      index;                   /* site index */
  int      p;                       /* velocity index */
  int      i, j;                    /* summed over indices ("alphabeta") */

  FVector  gphi;                    /* Gradient of phi */
  Float    grad_phi_sq;
  Float    jphi[3];                 /* Order parameter flux */
  Float    s[3][3];                 /* Stress */
  FTensor  sth;                     /* Equilibrium stress (thermodynamic) */
  Float    sphi[3][3];              /* Order parameter "stress" */
  Float    rho, phi;                /* Density, and order parameter */
  Float    *f,*g;

  Float    udotc;
  Float    jdotc;
  Float    sdotq, sphidotq;

  Float    s1;                      /* Diagonal part of thermodynamic stress */
  Float    A, B, kappa;             /* Free energy parameters */

  const Float r2     = 0.5;
  const Float c3r2   = 1.5;
  const Float rcs2   = 3.0;         /* The constant 1 / c_s^2 */
  const Float r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  Float    dp0;

  /* WARNING: phi_site[], and grad_phi[] must be up-to-date!!! */

  index = ps-site;

  /* JCD: Just in case (not suitable for colloidal systems... for now) */
  /* Needs to be tidied up */
  if(site_map[index] != FLUID){
    fatal("MODEL_set_distributions_atrest() for non-fluid site\n");
  }

  A     = free_energy_A();
  B     = free_energy_B();
  kappa = free_energy_K();
  get_N_local(N);

  f = ps->f;
  g = ps->g;

  /* Tables need to be up-to-date (see assertions at start) */
  rho = rho_site[index];
  phi = phi_site[index];

  jphi[0] = jphi[1] = jphi[2] = 0.0;
  for(p = 1; p < NVEL; p++){
    jphi[0] += g[p]*cv[p][0];
    jphi[1] += g[p]*cv[p][1];
    jphi[2] += g[p]*cv[p][2];
  }

  for(i = 0; i < 3; i++){
    for(j = 0; j < 3; j++){
      sphi[i][j] = 0.0;
      for(p = 1; p < NVEL; p++){
	sphi[i][j] += g[p]*_q[p][i][j];
      }
    }
  }

  /* 
   * Note: in the equivalent fragment of code in the collision (multirelaxation)
   * the momentum is normally scaled done by 1/rho at this stage. Here, it isn't
   */

  /* Thermodynamic part of the stress */
  gphi.x = (grad_phi + index)->x;
  gphi.y = (grad_phi + index)->y;
  gphi.z = (grad_phi + index)->z;

  grad_phi_sq = gphi.x*gphi.x + gphi.y*gphi.y + gphi.z*gphi.z;
  s1 = r2*phi*phi*(A + c3r2*B*phi*phi)
    - kappa*(phi*delsq_phi[index] + r2*grad_phi_sq);

  sth.x.x = s1 + kappa*gphi.x*gphi.x;
  sth.x.y =      kappa*gphi.x*gphi.y;
  sth.x.z =      kappa*gphi.x*gphi.z;
  sth.y.x = sth.x.y;
  sth.y.y = s1 + kappa*gphi.y*gphi.y;
  sth.y.z =      kappa*gphi.y*gphi.z;
  sth.z.x = sth.x.z;
  sth.z.y = sth.y.z;
  sth.z.z = s1 + kappa*gphi.z*gphi.z;

  /* Compute s[][] */
  /* JCD: note - sign (in -eta grad_u)... as suggested by Ronojoy */
  /* JCD: additional sth term required for 2-fluid (see Ronojoy) */
  s[0][0] = rho*u.x*u.x - eta_shear*grad_u.x.x + sth.x.x;
  s[0][1] = rho*u.x*u.y - eta_shear*grad_u.x.y + sth.x.y;
  s[0][2] = rho*u.x*u.z - eta_shear*grad_u.x.z + sth.x.z;
  s[1][0] = rho*u.y*u.x - eta_shear*grad_u.y.x + sth.y.x;
  s[1][1] = rho*u.y*u.y - eta_shear*grad_u.y.y + sth.y.y;
  s[1][2] = rho*u.y*u.z - eta_shear*grad_u.y.z + sth.y.z;
  s[2][0] = rho*u.z*u.x - eta_shear*grad_u.z.x + sth.z.x;
  s[2][1] = rho*u.z*u.y - eta_shear*grad_u.z.y + sth.z.y;
  s[2][2] = rho*u.z*u.z - eta_shear*grad_u.z.z + sth.z.z;
  
  /* Now update the distributions */
  for(p = 0; p < NVEL; p++){
    dp0 = (p == 0);

    udotc = u.x*cv[p][0] + u.y*cv[p][1] + u.z*cv[p][2];
    jdotc = jphi[0]*cv[p][0] + jphi[1]*cv[p][1] + jphi[2]*cv[p][2];

    sdotq    = 0.0;
    sphidotq = 0.0;
    for(i = 0; i < 3; i++){
      for(j = 0; j < 3; j++){
	sdotq += s[i][j]*_q[p][i][j];
	sphidotq += sphi[i][j]*_q[p][i][j];
      }
    }

    /* Project all this back to the distributions. The magic
     * here is to move phi into the non-propagating distribution. */
    f[p] = wv[p]*(rho + rho*udotc*rcs2 + sdotq*r2rcs4);
    g[p] = wv[p]*(          jdotc*rcs2 + sphidotq*r2rcs4) + phi*dp0;
  }

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
 *- \c See \c also: COM_init(), LE_init(), MODEL_process_options(), 
 *                COM_read_link(), MODEL_read_site_data()
 *- \c Note:      none
 */
/*----------------------------------------------------------------------------*/

void MODEL_init( void )
{
  int     i,j,k,ind,xfac,yfac,N_sites;
  int     N[3];
  int     offset[3];
  Float   phi;


  LUDWIG_ENTER("MODEL_init()");

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
  info("Requesting %d bytes for delsq_phi\n", N_sites*sizeof(Float));
  info("Requesting %d bytes for rho_site\n", N_sites*sizeof(Float));

  grad_phi  = (FVector*)calloc(N_sites,sizeof(FVector));
  delsq_phi = (Float  *)calloc(N_sites,sizeof(Float  ));
  rho_site  = (Float  *)calloc(N_sites,sizeof(Float  ));

  if(grad_phi==NULL || delsq_phi==NULL || rho_site==NULL)
    {
      fatal("MODEL_init(): failed to allocate %d bytes for vels\n",
	    2*N_sites*(sizeof(FVector)+sizeof(Float)));
    }

  LATT_allocate_sites(N_sites);
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

  /* Option 1: read distribution functions from file */
  if( strcmp(gbl.input_config,"EMPTY") != 0 ) {

    info("Re-starting simulation at step %d with data read from "
	 "config\nfile(s) %s\n", get_step(), gbl.input_config);

    /* Read distribution functions - sets both */
    COM_read_site(gbl.input_config,MODEL_read_site);
  } 
  /* Option 6: set rho/phi to defaults */
  else {
      /* 
       * Provides same initial conditions for rho/phi regardless of the
       * decomposition (all implementations, i.e. serial, MPI, OpenMP) 
       */
      
      /* Initialise lattice with constant density */
      /* Initialise phi with initial value +- noise */

      for(i=1; i<=N_total(X); i++)
	for(j=1; j<=N_total(Y); j++)
	  for(k=1; k<=N_total(Z); k++) {

	    ind = i*xfac + j*yfac + k;

	    phi = gbl.phi + gbl.noise*(RAN_uniform() - 0.5);
	    if(fabs(phi) < TINY){
	      phi = (phi > 0) ? TINY : (-TINY);
	    }

	    /* For computation with single fluid and no noise */
	    /* Only set values if within local box */
	    if((i>offset[X]) && (i<=offset[X] + N[X]) &&
	       (j>offset[Y]) && (j<=offset[Y] + N[Y]) &&
	       (k>offset[Z]) && (k<=offset[Z] + N[Z]))
	      {
		ind = (i-offset[X])*xfac + (j-offset[Y])*yfac + (k-offset[Z]);
		site_map[ind] = FLUID;
#ifdef _SINGLE_FLUID_
		MODEL_set_rho(gbl.rho, ind);
		MODEL_set_phi(gbl.phi, ind);
#else
		MODEL_set_rho(gbl.rho, ind); 
		MODEL_set_phi(phi,     ind);
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

/*----------------------------------------------------------------------------*/
/*!
 * Set simulation parameters to default values and parse user-supplied values
 * from input file
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: Input_Data *h: user-supplied parameter values
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   P. Bladon and JC Desplat
 *- \c See \c also: MODEL_init(), COM_read_input_file(), COM_process_options()
 *- \c Note:      although values for eta2/viscosity2 will be read from the
 *                input file, these parameters are not yet supported in the
 *                current release of the code
 */
/*----------------------------------------------------------------------------*/

void MODEL_process_options( Input_Data *h )
{
  Input_Data *p,*tmp;
  int  flag,i1;
  Float f1,f2;
  char parameter[256],value[256];
  
  LUDWIG_ENTER("MODEL_process_options()");

  /* Set defaults here:  D3Q15 parameters of Qian et. al. */
  gbl.rho = 1.0;
  gbl.phi = 0.0;
  gbl.mobility = 0.1;
  gbl.noise = 0.1;

  gbl.input_format = ASCII;
  gbl.output_format = ASCII;

  /* I/O */
  strcpy(gbl.input_config,"EMPTY");
  strcpy(gbl.output_config,"config.out");

  /* Read out list */
  p = h->next;
  while( p != NULL )
    {
      /* Get strings */
      if(sscanf(p->str,"%s %s",parameter,value) == EOF){
	sprintf(parameter,"INVALID_ENTRY");
      }

      flag = FALSE;
	
      /* Work out what we got */

      if( strcmp("input_format",parameter) == 0 )
	{
	  if( strncmp("BINARY",value,6) == 0 )
	    {
	      gbl.input_format = BINARY;
	      flag = TRUE;
	    }
	  else if( strncmp("ASCII",value,5) == 0 )
	    {
	      gbl.input_format = ASCII;
	      flag = TRUE;
	    }
	}
      else if( strcmp("output_format",parameter) == 0 )
	{
	  if( strncmp("BINARY",value,6) == 0 )
	    {
	      gbl.output_format = BINARY;
	      flag = TRUE;
	    }
	  else if( strncmp("ASCII",value,5) == 0 )
	    {
	      gbl.output_format = ASCII;
	      flag = TRUE;
	    }
	}
      else if( strcmp("rho",parameter) == 0 )
	{
	  gbl.rho = atof(value);
	  flag = TRUE;
	}
      else if( strcmp("phi",parameter) == 0 )
	{
	  gbl.phi = atof(value);
	  flag = TRUE;
	}
      else if( strcmp("mobility",parameter) == 0 )
	{
	  gbl.mobility = atof(value);
	  flag = TRUE;
	}
      else if( strcmp("noise",parameter) == 0 )
	{
	  gbl.noise = atof(value);
	  flag = TRUE;
	}
      else if( strcmp("input_config",parameter) == 0 )
	{
	  strcpy(gbl.input_config,value);
	  flag = TRUE;
	}
      else if( strcmp("output_config",parameter) == 0 )
	{
	  strcpy(gbl.output_config,value);
	  flag = TRUE;
	}	
      else if (strcmp("colloid_a0", parameter) == 0)
	{
	  Global_Colloid.a0 = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_ah", parameter) == 0)
	{
	  Global_Colloid.ah = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_vf", parameter) == 0)
	{
	  Global_Colloid.vf = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_r_lu_n", parameter) == 0)
	{
	  Global_Colloid.r_lu_n = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_r_lu_t", parameter) == 0)
	{ 
	  Global_Colloid.r_lu_t = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_r_lu_r", parameter) == 0)
	{
	  Global_Colloid.r_lu_r = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_r_ssph", parameter) == 0)
	{
	  Global_Colloid.r_ssph = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_r_clus", parameter) == 0)
	{
	  Global_Colloid.r_clus = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_drop_in_p1", parameter) == 0)
	{
	  Global_Colloid.drop_in_p1 = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_drop_in_p2", parameter) == 0)
	{
	  Global_Colloid.drop_in_p2 = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_drop_in_p3", parameter) == 0)
	{
	  Global_Colloid.drop_in_p3 = atof(value);
	  flag = TRUE;
	}
      else if (strcmp("colloid_gravity", parameter) == 0)
	{
	  if( sscanf(value,"%lg_%lg_%lg", &Global_Colloid.F.x,
		     &Global_Colloid.F.y, &Global_Colloid.F.z) != 3 )
	    {
	      fatal("colloid_gravity format not corrent in input\n");
	    }
	  flag = TRUE;
	}

      /* If got an option ok, remove from list */
      if( flag )
	{
	  if( p->next != NULL ) p->next->last = p->last;
	  if( p->last != NULL ) p->last->next = p->next;		
	  
	  tmp = p->next;
	  free(p);
	  p = tmp;
	}
      else
	{
	  /* Next String */
	  p = p->next;
	}
    }

  /* Set input routines: point to ASCII/binary routine depending on current 
     settings */
  switch( gbl.input_format )
    {
    case BINARY:
      MODEL_read_site     = MODEL_read_site_bin;
      break;

    case ASCII:
      MODEL_read_site     = MODEL_read_site_asc;
      break;
    }

  /* Set output routines: point to ASCII/binary routine depending on current 
     settings */
  switch( gbl.output_format )
    {
    case BINARY:
      MODEL_write_site     = MODEL_write_site_bin;
      MODEL_write_velocity = MODEL_write_velocity_bin;
      MODEL_write_rho      = MODEL_write_rho_bin;
      MODEL_write_phi      = MODEL_write_phi_bin;
      MODEL_write_rho_phi  = MODEL_write_rho_phi_bin;	
      break;

    case ASCII:
      MODEL_write_site     = MODEL_write_site_asc;
      MODEL_write_velocity = MODEL_write_velocity_asc;
      MODEL_write_rho      = MODEL_write_rho_asc;
      MODEL_write_phi      = MODEL_write_phi_asc;
      MODEL_write_rho_phi  = MODEL_write_rho_phi_asc;	
      break;
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Reads distribution function into site s from stream fp (ASCII input)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: FILE *fp: pointer for stream
 *- \c Returns:   void
 *- \c Buffers:   invalidates .halo, .rho, .phi, .grad_phi
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_read_site(), MODEL_read_site_bin(), COM_read_site()
 *- \c Note:      All buffers will need to be re-computed following a
 *                configuration read. An optimised version of this routine will
 *                need to be developed for the typical situation where all sites
 *                are read (simulation re-start). This may considerably speed-up
 *                this operation depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_read_site_asc( FILE *fp )
{
  int i,ind,g_ind;
  
  LUDWIG_ENTER("MODEL_read_site_asc()");

  if( fscanf(fp,"%d",&g_ind) != EOF )
    {
      /* Get local index */
      ind = COM_local_index( g_ind );

      for(i=0; i<NVEL; i++)
	{
	  if( fscanf(fp,"%lg %lg",&site[ind].f[i],&site[ind].g[i]) == EOF )
	    {
	      fatal("MODEL_read_site_asc(): read EOF\n");
	    }
	}
    }
  else
    {
      fatal("MODEL_read_site_asc(): read EOF\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Reads distribution function into site s from stream fp (binary input)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: FILE *fp: pointer for stream
 *- \c Returns:   void
 *- \c Buffers:   invalidates .halo, .rho, .phi, .grad_phi
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_read_site(), MODEL_read_site_asc(), COM_read_site()
 *- \c Note:      All buffers will need to be re-computed following a
 *                configuration read. An optimised version of this routine will
 *                need to be developed for the typical situation where all sites
 *                are read (simulation re-start). This may considerably speed-up
 *                this operation depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_read_site_bin( FILE *fp )
{
  int ind,g_ind;
  
  LUDWIG_ENTER("MODEL_read_site_bin()");

  if( fread(&g_ind,sizeof(int),1,fp) != 1 )
    {
      fatal("MODEL_read_site_bin(): couldn't read index\n");
    }
  
  /* Convert to local index */
  ind = COM_local_index( g_ind );
  
  if( fread(site+ind,sizeof(Site),1,fp) != 1 )
    {
      fatal("MODEL_read_site_bin(): couldn't read site\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes distribution function from site s with index ind to stream fp (ASCII
 * output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_site(), MODEL_write_site_bin(), COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written (configuration dump). This may considerably speed-up
 *                this operation depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_site_asc( FILE *fp, int ind, int g_ind )
{
  int i;

  LUDWIG_ENTER("MODEL_write_site_asc()");

  fprintf(fp,"%d ",g_ind);

  /* Write site information to stream */
  for(i=0; i<NVEL; i++)
    {
      if( fprintf(fp,"%g %g ",site[ind].f[i],site[ind].g[i]) < 0 )
	{
	  fatal("MODEL_write_site_asc(): couldn't write site data\n");
	}
    }
  fprintf(fp,"\n");
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes velocity data from site s with index ind to stream fp (ASCII output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_velocity(), MODEL_write_velocity_bin(), 
 *                COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written. This may considerably speed-up this operation 
 *                depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_velocity_asc( FILE *fp, int ind, int g_ind )
{
  FVector u;

  LUDWIG_ENTER("MODEL_write_velocity_asc()");

  u = MODEL_get_momentum_at_site(ind);

  /* Write velocity information to stream */
  if( fprintf(fp,"%d %lg %lg %lg\n",g_ind,u.x,u.y,u.z) < 0 )
    {
      fatal("MODEL_write_velocity_asc(): couldn't write velocity data\n");
    }

}

/*----------------------------------------------------------------------------*/
/*!
 * Writes density (rho) from site s with index ind to stream fp (ASCII output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_rho(), MODEL_write_rho_bin(), COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written. This may considerably speed-up this operation 
 *                depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_rho_asc( FILE *fp, int ind, int g_ind )
{
  Float rho;    

  LUDWIG_ENTER("MODEL_write_rho_asc()");

  rho = MODEL_get_rho_at_site(ind);

  /* Write density information to stream */
  if( fprintf(fp,"%d %lg\n",g_ind,rho) < 0 )
    {
      fatal("MODEL_write_rho_asc(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes composition (phi) from site s with index ind to stream fp (ASCII 
 * output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_phi(), MODEL_write_phi_bin(), COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written. This may considerably speed-up this operation
 *                depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_phi_asc( FILE *fp, int ind, int g_ind )
{
  Float phi;    

  LUDWIG_ENTER("MODEL_write_phi_asc()");

  phi = MODEL_get_phi_at_site(ind);

  /* Write composition information to stream */
  if( fprintf(fp,"%d %lg\n",g_ind,phi) < 0 )
    {
      fatal("MODEL_write_phi_asc(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes density (rho) and composition (phi) data from site s with index ind
 * to stream fp (ASCII output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_rho_phi(), MODEL_write_rho_phi_bin(), 
 *                COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written. This may considerably speed-up this operation 
 *                depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_rho_phi_asc( FILE *fp, int ind, int g_ind )
{
  Float rho, phi;    

  LUDWIG_ENTER("MODEL_write_rho_phi_asc()");

  rho = MODEL_get_rho_at_site(ind);
  phi = MODEL_get_phi_at_site(ind);

  /* Write density and composition information to stream */
  if( fprintf(fp,"%d %lg %lg\n",g_ind,rho,phi) < 0 )
    {
      fatal("MODEL_write_rho_phi_asc(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes distribution function from site s with index ind to stream fp (binary
 * output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_site(), MODEL_write_site_asc(), COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written (configuration dump). This may considerably speed-up
 *                this operation depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_site_bin( FILE *fp, int ind, int g_ind )
{
  LUDWIG_ENTER("MODEL_write_site_bin()");

  if( fwrite(&g_ind,sizeof(int),1,fp)    != 1  ||
      fwrite(site+ind,sizeof(Site),1,fp) != 1 )
    {
      fatal("MODEL_write_site_bin(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes velocity data from site s with index ind to stream fp (binary output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_velocity(), MODEL_write_velocity_asc(), 
 *                COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written. This may considerably speed-up this operation 
 *                depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_velocity_bin( FILE *fp, int ind, int g_ind )
{
  FVector u;

  LUDWIG_ENTER("MODEL_write_velocity_bin()");

  u = MODEL_get_momentum_at_site(ind);

  if( fwrite(&g_ind,sizeof(int),1,fp) != 1 ||
      fwrite(&u.x,sizeof(Float),1,fp) != 1  ||
      fwrite(&u.y,sizeof(Float),1,fp) != 1  ||
      fwrite(&u.z,sizeof(Float),1,fp) != 1  )
    {
      fatal("MODEL_write_velocity_bin(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes density (rho) from site s with index ind to stream fp (binary output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_rho(), MODEL_write_rho_asc(), COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written. This may considerably speed-up this operation 
 *                depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_rho_bin( FILE *fp, int ind, int g_ind )
{
  Float rho;    

  LUDWIG_ENTER("MODEL_write_rho_bin()");

  rho = MODEL_get_rho_at_site(ind);

  if( fwrite(&g_ind,sizeof(int),1,fp) != 1 ||
      fwrite(&rho,sizeof(Float),1,fp) != 1 )
    {
      fatal("MODEL_write_rho_bin(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes composition (phi) from site s with index ind to stream fp (binary
 * output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_phi(), MODEL_write_phi_asc(), COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written. This may considerably speed-up this operation
 *                depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_phi_bin( FILE *fp, int ind, int g_ind )
{
  Float phi;    

  LUDWIG_ENTER("MODEL_write_phi_bin()");

  phi = MODEL_get_phi_at_site(ind);

  if( fwrite(&g_ind,sizeof(int),1,fp) != 1 ||
      fwrite(&phi,sizeof(Float),1,fp) != 1 )
    {
      fatal("MODEL_write_phi_bin(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes density (rho) and composition (phi) data from site s with index ind
 * to stream fp (binary output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_rho_phi(), MODEL_write_rho_phi_asc(), 
 *                COM_write_site()
 *- \c Note:      An optimised version of this routine will need to be
 *                developed for the typical situation where all sites are
 *                written. This may considerably speed-up this operation 
 *                depending on how the OS manages disc buffers
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_rho_phi_bin( FILE *fp, int ind, int g_ind )
{
  Float rho,phi;    

  LUDWIG_ENTER("MODEL_write_rho_phi_bin()");

  rho = MODEL_get_rho_at_site(ind);
  phi = MODEL_get_phi_at_site(ind);

  if( fwrite(&g_ind,sizeof(int),1,fp) != 1 ||
      fwrite(&rho,sizeof(Float),1,fp) != 1 ||
      fwrite(&phi,sizeof(Float),1,fp) != 1 )
    {
      fatal("MODEL_write_rho_phi_bin(): couldn't write data\n");
    }
}

/*****************************************************************************
 *
 *  MODEL_set_rho
 *
 *  Project rho onto the distribution at position index, assuming zero
 *  velocity and zero stress.
 *
 *****************************************************************************/

void MODEL_set_rho(const double rho, const int index) {

  int   p;

  for (p = 0; p < NVEL; p++) {
    site[index].f[p] = wv[p]*rho;
  }

  return;
}

/*****************************************************************************
 *
 *  MODEL_set_phi
 *
 *  Sets the order parameter distribution at index address, assuming
 *  zero order parameter flux and zero stress.
 *
 *  Note that this is currently incompatible with the reprojection
 *  at the collision stage where all the phi would go into the rest
 *  distribution.
 *
 ****************************************************************************/

void MODEL_set_phi(const double phi, const int index) {

  int   p;

  for (p = 0; p < NVEL; p++) {
    site[index].g[p] = wv[p]*phi;
  }

  return;
}

/*****************************************************************************
 *
 *  MODEL_get_rho_at_site
 *
 *  Return the density at lattice node index.
 *
 *****************************************************************************/

double MODEL_get_rho_at_site(const int index) {

  double rho;
  double * f;
  int   p;

  rho = 0.0;
  f = site[index].f;

  for (p = 0; p < NVEL; p++)
    rho += f[p];

  return rho;
}

/****************************************************************************
 *
 *  MODEL_get_phi_at_site
 *
 *  Return the order parameter at lattice node index.
 *
 ****************************************************************************/

double MODEL_get_phi_at_site(const int index) {

  double   phi;
  double * g;
  int     p;

  phi = 0.0;
  g = site[index].g;

  for (p = 0; p < NVEL; p++) {
    phi += g[p];
  }

  return phi;
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
  Float *f;

  
  LUDWIG_ENTER("MODEL_calc_rho()");
  
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

double normalise = 0.0;

void RAND_init_fluctuations() {

  int  i, j, p;
  double tau;

  /* Initialise the viscosities and relaxation times */

  p = RUN_get_double_parameter("viscosity", &eta_shear);

  if (p == 0) {
    /* Default relaxation time of unity...  */
    eta_shear = 1.0/6.0;
    eta_bulk  = eta_shear;
  }
  else {
    RUN_get_double_parameter("viscosity_bulk", &eta_bulk);
  }

  rtau_shear = 2.0 / (1.0 + 6.0*eta_shear);
  rtau_bulk  = 2.0 / (1.0 + 6.0*eta_bulk);

  tau = 1.0/rtau_shear;

  p = RUN_get_double_parameter("temperature", &normalise);

  /* Initialise q matrix */

  var_shear = normalise*sqrt(2.0*2.0*eta_shear/(3.0*tau*tau));
  var_bulk  = var_shear/sqrt(2.0);

  for (p = 0; p < NVEL; p++) {

    for (i = 0; i < 3; i++) {

      for (j = 0; j < 3; j++) {
	if (i == j) {
	  _q[p][i][j] = cv[p][i]*cv[p][j] - 1.0/3.0;
	}
	else {
	  _q[p][i][j] = cv[p][i]*cv[p][j];
	}
      }
    }
  }

  init_ghosts(normalise);

  /* Collision global force on fluid defaults to zero. */

  siteforce[X] = 0.0;
  siteforce[Y] = 0.0;
  siteforce[Z] = 0.0;

  p = RUN_get_double_parameter_vector("force", siteforce);

  return;
}


/*****************************************************************************
 *
 *  RAND_fluctuations
 *
 *  Thermal fluctuations following Ronojoy.
 *
 *  This routine computes a vector fr[] of fluctuations which are
 *  added to the denisty distribution function after the collision
 *  stage.
 *
 *  A set of variances are required which depend upon random numbers
 *  and pre-computed factors.
 *
 *****************************************************************************/

void get_fluctuations_stress(double shat[3][3]) {

  shat[0][0] = var_shear*ran_parallel_gaussian();
  shat[1][1] = var_shear*ran_parallel_gaussian();
  shat[2][2] = var_shear*ran_parallel_gaussian();

  shat[0][1] = var_bulk*ran_parallel_gaussian();
  shat[1][2] = var_bulk*ran_parallel_gaussian();
  shat[2][0] = var_bulk*ran_parallel_gaussian();

  shat[1][0] = shat[0][1];
  shat[2][1] = shat[1][2];
  shat[0][2] = shat[2][0];

  return;
}

void get_fluctuations_ghost(double fr[]) {
  
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

  return;
}

/*****************************************************************************
 *
 *  get_eta_shear
 *
 *  Return the shear viscosity.
 *
 *****************************************************************************/

double get_eta_shear() {

  return eta_shear;
}
