/*****************************************************************************
 *
 *  wall.c
 *
 *  Routines for simple solid boundary walls.
 *
 *  These walls may be used to impart sheer to the system (or just
 *  break periodicity) and are designed to work with colloidal
 *  particles.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "globals.h"

#include "pe.h"
#include "coords.h"
#include "cartesian.h"

/* Global quantities */

Wall        _wall;

static Wall_link * WALL_allocate_wall_link(void);
static void        WALL_init_fluid(void);
static void        WALL_init_side_wall_links(void);
static void        WALL_init_site_map(void);


/*****************************************************************************
 *
 *  WALL_init
 *
 *  Create a boundary wall with the specified parameters
 *  (hard-wired in at the moment).
 *
 *  This wall is designed to take up two planes in a given
 *  coordinate direction, one at the 'bottom' and one at the 'top'.
 *  The walls can be given a velocity in their own plane to impart
 *  sheer to the fluid. The 'top' and 'bottom' plane are independent
 *  and so can move with different velocities.
 *
 *  Oscillatory sheer is built in if required.
 *
 ****************************************************************************/

void WALL_init() {

  if (! _wall.present) {
    /* Peridic in all three directions */
  }
  else {

    _wall.orientation = WALL_XY;             /* and only XY at the moment */
    _wall.rlower = 1.0;                      /* bottom */
    _wall.rupper = (Float) N_total(Z);       /* top */
    _wall.C = 0.0;
    _wall.H = 0.0;
    _wall.lnkupper = NULL;
    _wall.lnklower = NULL;

    WALL_init_side_wall_links();
    WALL_update(0);
    WALL_init_site_map();

    /* Switch off periodic boundaries in the z-direction */

    WALL_update(0);
    WALL_init_fluid();

    info("WALL_init_side_wall:\n");
    info("Initialised side walls in XY plane\n");
    info("sites at     = (z = %f and z = %f)\n", _wall.rlower, _wall.rupper);
    info("lubrication  = %f\n", _wall.r_lu_n);
    info("max velocity = (%f, %f)\n", _wall.sheer_uxmax, _wall.sheer_uymax);
    info("sheer diff   = %f\n", _wall.sheer_diff);
    info("sheer period = %f\n", _wall.sheer_period);
 
 }

  return;
}


/*****************************************************************************
 *
 *  WALL_init_fluid()
 *
 *  Look at the initial wall velocity and set an initial shear
 *  profile in the fluid to match. A simple linear profile is
 *  assumed; if the walls are initially stationary, nothing is
 *  done.
 *
 *  Will need to be extended to binary fluid. Single fluid only
 *  at the moment.
 *
 *  The z-coordinate is slightly oscure when walls are present:
 *  remember the first fluid node is at z = 2.
 *
 *****************************************************************************/

void WALL_init_fluid() {
  
  int     ic, jc, kc, index, xfac, yfac;
  int     p;
  int     N[3];
  double   uxsheer, uysheer;
  double * f;
  double   udotc;
  double   uxbottom, uybottom, dux, duy;

  const   double rho  = 1.0;
  const   double rcs2 = 3.0;

  get_N_local(N);
  xfac     = (N[Y]+2)*(N[Z]+2);
  yfac     = (N[Z]+2);

  /* Work out the sheer between top and bottom */
  uxbottom = _wall.sheer_u.x*_wall.sheer_diff;
  uybottom = _wall.sheer_u.y*_wall.sheer_diff;

  dux = (_wall.sheer_u.x - uxbottom) / (double) (N[Z] - 2.0);
  duy = (_wall.sheer_u.y - uybottom) / (double) (N[Z] - 2.0);


  /* Set up the initial distributions */

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 2; kc < N[Z]; kc++) {

	index = xfac*ic + yfac*jc + kc;

	f = site[index].f;

	/* Interpolate the wall velocity between top and bottom */
	/* The extra 0.5 allows the velocity to match at the
	 * surface of the wall (bounce-back half way) */

	/* linear */
	uxsheer = uxbottom + dux*(kc - 1.5);
	uysheer = uybottom + duy*(kc - 1.5);

	for (p = 0; p < NVEL; p++) {
	  udotc = cv[p][0]*uxsheer + cv[p][1]*uysheer;
	  f[p] = wv[p]*rho*(1.0 + rcs2*udotc);
	}

      }
    }
  }

  return;
}


/*****************************************************************************
 *
 *  WALL_init_side_wall_links
 *
 *  Initialise the links for a side wall. It is envisaged that the
 *  side wall will not move its position, so this need only be
 *  called once at start of execution.
 *
 ****************************************************************************/

void WALL_init_side_wall_links() {

  int         i, j, k, index, index1;
  int         xfac, yfac;
  int         p;

  Wall_link * tmp;

  yfac = (N_total(Z) + 2);
  xfac = (N_total(Y) + 2)*yfac;

  /* Loop through sites in the domain and in the XY plane */

  for (i = 1; i <= N_total(X); i++)
    for (j = 1; j <= N_total(Y); j++) {

      /* At the lower side */
      k     = 2;
      index = xfac*i + yfac*j + k;

      /* Add links joining k = 2 with k = 1 and set the appropriate
       * properties */

      for (p = 1; p < NVEL; p++) {

	if (cv[p][2] == -1) {
	  tmp = WALL_allocate_wall_link();

	  index1 = index + xfac*cv[p][0] + yfac*cv[p][1] - 1;
	  tmp->i = index;
	  tmp->j = index1;
	  tmp->p = p;

	  tmp->next = _wall.lnklower;
	  _wall.lnklower = tmp;
	}

      }

      /* At the upper side */
      k = N_total(Z) - 1;
      index = xfac*i + yfac*j + k;

      /* Add links joining the penultimate site with sites at k = N.z
       * and set appropriate properties. */

      for (p = 1; p < NVEL; p++) {

	if (cv[p][2] == 1) {
	  tmp = WALL_allocate_wall_link();

	  index1 = index + xfac*cv[p][0] + yfac*cv[p][1] + 1;
	  tmp->i = index;
	  tmp->j = index1;
	  tmp->p = p;

	  tmp->next = _wall.lnkupper;
	  _wall.lnkupper = tmp;
	}
      }

    }

  return;
}


/*****************************************************************************
 *
 *  WALL_init_site_map
 *
 *  Set the site map to SOLID for the boundary walls.
 *
 *  Issues
 *    It is envisaged that the walls will not change position
 *    for the duration of the run, i.e., they only move parallel
 *    to their plane.
 *
 *****************************************************************************/

void WALL_init_site_map() {

  int      i, j, index;
  int      xfac, yfac;

  yfac = (N_total(Z) + 2);
  xfac = (N_total(Y) + 2)*yfac;

  for (i = 0; i <= N_total(X) +1; i++)
    for (j = 0; j <= N_total(Y) +1; j++) {

      index = xfac*i + yfac*j;

      site_map[index + 1]   = SOLID;
      site_map[index + N_total(Z)] = SOLID;
    }

  return;
}


/*****************************************************************************
 *
 *  WALL_update
 *
 *  It is envisaged that this function is called once per time step
 *  to set the current sheer velocity of the wall.
 * 
 *  And in any case at least once to set the wall velocity at the
 *  start of the model run.
 *
 *****************************************************************************/

void WALL_update(int step) {

  double phase;
  double udotc;

  int      i, j, index, p;
  int      xfac, yfac;

  if (! _wall.present) {
    /* Do nothing */
  }
  else {

    if (_wall.sheer_period <= 0.0) {
      /* Time independent sheer velocity */

      _wall.sheer_u.x = _wall.sheer_uxmax;
      _wall.sheer_u.y = _wall.sheer_uymax;
      _wall.sheer_u.z = 0.0;

    }
    else {
      /* Sinusiodally varying sheer */

      phase = sin(2.0*PI*step/_wall.sheer_period);

      _wall.sheer_u.x = _wall.sheer_uxmax*phase;
      _wall.sheer_u.y = _wall.sheer_uymax*phase;
      _wall.sheer_u.z = 0.0;
    }

    yfac = (N_total(Z) + 2);
    xfac = (N_total(Y) + 2)*yfac;

    for (i = 0; i <= N_total(X) + 1; i++)
      for (j = 0; j <= N_total(Y)+1; j++) {

	index = xfac*i + yfac*j;

	for (p = 0; p < NVEL; p++) {
	  udotc = 3.0*(_wall.sheer_u.x*cv[p][0] + _wall.sheer_u.y*cv[p][1]);

	  site[index + 1].f[p] = wv[p]*(1.0 + _wall.sheer_diff*udotc);
	  site[index + N_total(Z)].f[p] = wv[p]*(1.0 + udotc);
#ifdef _DUCT_
	  site[index + N_total(Z)-1].f[p] = wv[p]*(1.0 + udotc);
	  /*
	  udotc = 3.0*ubot*cv[p][0]; 
	  site[index + 2].f[p] = wv[p]*(1.0 + udotc);*/
#endif
	}
      }
  }

  return;
}


/****************************************************************************
 *
 *  WALL_allocate_wall_link
 *
 *  Return a pointer to a newly allocated wall boundary link structure
 *  or fail gracefully.
 *
 ****************************************************************************/

Wall_link * WALL_allocate_wall_link() {

  Wall_link * p_link;

  p_link = (Wall_link *) malloc(sizeof(Wall_link));

  if (p_link == (Wall_link *) NULL) {
    fatal("malloc(Wall_link) failed (requested %d bytes)\n",
	  sizeof(Wall_link));
  }

  return p_link;
}

/*****************************************************************************
 *
 *  WALL_bounce_back
 *
 *  Bounce back on links for the wall.
 *  To be called between after the collision step.
 *
 *  Issues:
 *    No OMP at the moment (use at most two threads?).
 *
 *****************************************************************************/

void WALL_bounce_back() {

  Wall_link * p_link;
  int         i, j, ij, ji;
  double       rho;
  double       cdotu, dtmp;

  rho = gbl.rho;

  p_link = _wall.lnklower;

  while (p_link) {

    i  = p_link->i;
    j  = p_link->j;
    ij = p_link->p;
    ji = NVEL - ij;

    cdotu = cv[ij][0]*_wall.sheer_u.x + cv[ij][1]*_wall.sheer_u.y;
    cdotu = _wall.sheer_diff*cdotu;
    dtmp = 2.0*wv[ij]*cdotu/3.0;

    site[j].f[ji] = site[i].f[ij] - dtmp*rho;

#ifdef _SINGLE_FLUID_
#else
    /* For phi should generally use site value */
    /*    site[j].g[ji] = site[i].g[ij] - dtmp*phi_site[i];*/
    site[j].g[ji] = site[i].g[ij] - dtmp*p_link->phi_b;
#endif

    p_link = p_link->next;
  }

  /* Repeat for the upper side */

  p_link = _wall.lnkupper;

  while (p_link) {

    i  = p_link->i;
    j  = p_link->j;
    ij = p_link->p;
    ji = NVEL -ij;

    cdotu = cv[ij][0]*_wall.sheer_u.x + cv[ij][1]*_wall.sheer_u.y;
    dtmp = 2.0*wv[ij]*cdotu/3.0;

    site[j].f[ji] = site[i].f[ij] - dtmp*rho;

#ifdef _SINGLE_FLUID_
#else
    /* For phi should generally use site value */
    /*    site[j].g[ji] = site[i].g[ij] - dtmp*phi_site[i];*/
    site[j].g[ji] = site[i].g[ij] - dtmp*p_link->phi_b;
#endif

    p_link = p_link->next;
  }

  return;
}


/******************************************************************************
 *
 *  WALL_lubrication
 *
 *  For a given colloid, add any appropriate particle-wall lubrication
 *  force. Again, this relies on the fact that the walls have no
 *  component of velocity nornal to their own plane.
 *
 *  The result should be added to the appropriate diagonal element of
 *  the colloid's drag matrix in the implicit update.
 *
 *  Issues
 *    Normal force is added to the diagonal of drag matrix \zeta^FU_zz
 *    Tangential force to \zeta^FU_xx and \zeta^FU_yy
 *
 *    If the colloid is near to both the top and bottom walls the system
 *    is too narrow!
 *
 *    Again, assumes wall is in XY plane.
 *
 *****************************************************************************/

FVector WALL_lubrication(Colloid * p_colloid) {

  FVector fl;
  double ah;
  double r_lu_n, r_lu_t;
  double gap, s, s0;

  fl = UTIL_fvector_zero();

  if (! _wall.present) {
    /* Do nothing */
  }
  else {
    ah = p_colloid->ah;
    r_lu_n = Global_Colloid.r_lu_n;
    r_lu_t = Global_Colloid.r_lu_t;

    /* Lower wall */

    gap = p_colloid->r.z - _wall.rlower;
    gap = gap - ah - _wall.r_lu_n;

    if (gap < 0.0) {
      verbose("---> WALL_lubrication:\n");
      verbose("---> Particle %d overlapped lower wall\n", p_colloid->index);
      verbose("---> Particle position: (%g, %g, %g)\n",
	      p_colloid->r.x, p_colloid->r.y, p_colloid->r.z);
      verbose("---> Particle velocity: (%g, %g, %g)\n",
	      p_colloid->v.x, p_colloid->v.y, p_colloid->v.z);
      fatal("aborted\n");
    }

    /* Normal force */
    if (gap < r_lu_n) {
      fl.z = -6.0*PI*get_eta_shear()*ah*ah*(1.0/gap - 1.0/r_lu_n);
    }

    /* Tangential force (dependent on particle velocity only) */
    if (gap < r_lu_t) {
      s  = log(gap/(gap + ah));
      s0 = log(r_lu_t/(r_lu_t + ah));
      fl.x = +6.0*PI*get_eta_shear()*ah*(s - s0);
      fl.y = +6.0*PI*get_eta_shear()*ah*(s - s0);
    }

    /* Upper wall */

    gap = _wall.rupper - p_colloid->r.z;
    gap = gap - ah - _wall.r_lu_n;

    if (gap < 0.0) {
      verbose("---> WALL_lubrication:\n");
      verbose("---> Particle %d overlapped upper wall\n", p_colloid->index);
      verbose("---> Particle position: (%g, %g, %g)\n",
	      p_colloid->r.x, p_colloid->r.y, p_colloid->r.z);
      verbose("---> Particle velocity: (%g, %g, %g)\n",
	      p_colloid->v.x, p_colloid->v.y, p_colloid->v.z);
      fatal("aborted\n");
    }

    if (gap < r_lu_n) {
      fl.z = -6.0*PI*get_eta_shear()*ah*ah*(1.0/gap - 1.0/r_lu_n);
    }
    if (gap < r_lu_t) {
      s  = log(gap/(gap + ah));
      s0 = log(r_lu_t/(r_lu_t + ah));
      fl.x = +6.0*PI*get_eta_shear()*ah*(s - s0);
      fl.y = +6.0*PI*get_eta_shear()*ah*(s - s0);
    }
  }

  return fl;
}
