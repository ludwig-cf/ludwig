/*****************************************************************************
 *
 *  wall.c
 *
 *  Static solid objects.
 *
 *  Special case: boundary walls.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "model.h"
#include "wall.h"
#include "site_map.h"
#include "runtime.h"

typedef struct B_link_struct B_link;

struct B_link_struct {

  int         i;     /* Outside (fluid) lattice node index */
  int         j;     /* Inside (solid) lattice node index */
  int         p;     /* Basis vector for this link */
  double     ux;     /* x-component boundary speed */

  B_link * next;     /* This is a linked list */
};

static int nalloc_links_ = 0;       /* Current number of links allocated */
static B_link * link_list_ = NULL;  /* Boundary links. */
static int is_boundary_wall_ = 0;   /* Any boundaries present? */
static double fnet_[3];             /* Accumulated net force on walls */

static B_link * allocate_link(void);
static void     init_links(void);
static void     init_boundary_site_map(void);
static void     init_boundary_speeds(const double, const double);
static void     set_wall_velocity(void);
static void     wall_shear_init(double utop, double ubottom);

/*****************************************************************************
 *
 *  wall_init
 *
 ****************************************************************************/

void wall_init(void) {

  int init_shear = 0;
  int ntotal;
  double ux_bottom = 0.0;
  double ux_top = 0.0;
  char   tmp[128];

  RUN_get_double_parameter("boundary_speed_bottom", &ux_bottom);
  RUN_get_double_parameter("boundary_speed_top", &ux_top);
  RUN_get_string_parameter("boundary_walls_on", tmp, 128);

  if (strcmp(tmp, "yes") == 0) is_boundary_wall_ = 1;

  if (is_boundary_wall_) {
    info("\n");
    info("Boundary walls\n");
    info("--------------\n");
    init_boundary_site_map();
    init_links();
    init_boundary_speeds(ux_bottom, ux_top);

    RUN_get_int_parameter("boundary_shear_init", &init_shear);
    if (init_shear) wall_shear_init(ux_top, ux_bottom);

    info("Boundary walls:                %1s %1s %1s\n",
	 (is_periodic(X) == 0) ? "X" : "-",
	 (is_periodic(Y) == 0) ? "Y" : "-",
	 (is_periodic(Z) == 0) ? "Z" : "-");
    info("Boundary speed u_x (bottom):  %14.7e\n", ux_bottom);
    info("Boundary speed u_x (top):     %14.7e\n", ux_top);
    info("Boundary shear initialise:     %d\n", init_shear);

    MPI_Reduce(&nalloc_links_, &ntotal, 1, MPI_INT, MPI_SUM, 0, pe_comm());
    info("Boundary links allocated:      %d\n", ntotal);
    info("Memory (total, bytes):         %d\n", ntotal*sizeof(B_link));
  }

  fnet_[X] = 0.0;
  fnet_[Y] = 0.0;
  fnet_[Z] = 0.0;

  return;
}

/*****************************************************************************
 *
 *  wall_present
 *
 *  Return 0 if no boundaries are present, or 1 if any boundary
 *  at all.
 *
 *****************************************************************************/

int wall_present(void) {
  return is_boundary_wall_;
}

/*****************************************************************************
 *
 *  wall_at_edge
 *
 *  Return 1 if there is a wall in the given direction.
 *
 *  At the moment, this information is implicit in the periodicity of
 *  the Cartesian communicator; it would be better to have it explicit
 *  (from input).
 *
 *****************************************************************************/

int wall_at_edge(const int d) {

  assert(d == X || d == Y || d == Z);

  return (is_boundary_wall_ && !is_periodic(d));
}

/*****************************************************************************
 *
 *  wall_update
 *
 *  Call once per time step to provide any update to wall parameters.
 *
 *****************************************************************************/

void wall_update() {

  set_wall_velocity();

  return;
}

/*****************************************************************************
 *
 *  wall_finish
 *
 *****************************************************************************/

void wall_finish() {

  B_link * p_link;
  B_link * p_tmp;

  p_link = link_list_;

  while (p_link) {
    p_tmp = p_link->next;
    free(p_link);
    nalloc_links_--;
    p_link = p_tmp;
  }

  return;
}

/*****************************************************************************
 *
 *  wall_bounce_back
 *
 *  Bounce back each distribution.
 *
 *****************************************************************************/

void wall_bounce_back(void) {

  B_link * p_link;
  int      i, j, ij, ji, ia;
  int      n, ndist;
  double   rho, cdotu;
  double   fp;
  double   force;

  p_link = link_list_;
  ndist = distribution_ndist();

  while (p_link) {

    i  = p_link->i;
    j  = p_link->j;
    ij = p_link->p;   /* Link index direction solid->fluid */
    ji = NVEL - ij;   /* Opposite direction index */

    cdotu = cv[ij][X]*p_link->ux;

    for (n = 0; n < ndist; n++) {

      fp = distribution_f(i, ij, n);
      rho = distribution_zeroth_moment(i, n);

      if (n == 0) {
	/* This is the momentum */
	force = 2.0*fp - 2.0*rcs2*wv[ij]*rho*cdotu;
	for (ia = 0; ia < 3; ia++) {
	  fnet_[ia] += force*cv[ij][ia];
	}
      }

      fp = fp - 2.0*rcs2*wv[ij]*rho*cdotu;
      distribution_f_set(j, ji, n, fp);

    }

    p_link = p_link->next;
  }

  return;
}

/*****************************************************************************
 *
 *  init_links
 *
 *  Look at the site map to determine fluid (strictly, non-solid)
 *  to solid links. Set once at the start of execution.
 *
 ****************************************************************************/

static void init_links() {

  int ic, jc, kc;
  int ic1, jc1, kc1, p;
  int n[3];

  B_link * tmp;

  coords_nlocal(n);

  for (ic = 1; ic <= n[X]; ic++) {
    for (jc = 1; jc <= n[Y]; jc++) {
      for (kc = 1; kc <= n[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;

	/* Look for non-solid -> solid links */

	for (p = 1; p < NVEL; p++) {

	  ic1 = ic + cv[p][X];
	  jc1 = jc + cv[p][Y];
	  kc1 = kc + cv[p][Z];

	  if (site_map_get_status(ic1, jc1, kc1) == BOUNDARY) {

	    /* Add a link to head of the list */

	    tmp = allocate_link();
	    tmp->i = coords_index(ic, jc, kc);        /* fluid site */
	    tmp->j = coords_index(ic1, jc1, kc1);     /* solid site */
	    tmp->p = p;
	    tmp->ux = 0.0;

	    tmp->next = link_list_;
	    link_list_ = tmp;
	  }
	  /* Next p. */
	}
	/* Next site */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  init_boundary_site_map
 *
 *  Set the site map to BOUNDARY for the boundary walls.
 *
 *****************************************************************************/

static void init_boundary_site_map() {

  int ic, jc, kc;
  int ic_global, jc_global, kc_global;
  int nlocal[3];
  int noffset[3];
  int nextra;

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  nextra = coords_nhalo();

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	/* If this is an appropriate periodic boundary, set to solid */

	ic_global = ic + noffset[X];
	jc_global = jc + noffset[Y];
	kc_global = kc + noffset[Z];

	if (!is_periodic(Z)) {
	  if (kc_global == 0 || kc_global == N_total(Z) + 1) {
	    site_map_set_status(ic, jc, kc, BOUNDARY);
	  }
	}

	if (!is_periodic(Y)) {
	  if (jc_global == 0 || jc_global == N_total(Y) + 1) {
	    site_map_set_status(ic, jc, kc, BOUNDARY);
	  }
	}

	if (!is_periodic(X)) {
	  if (ic_global == 0 || ic_global == N_total(X) + 1) {
	    site_map_set_status(ic, jc, kc, BOUNDARY);
	  }
	}
	/* next site */
      }
    }
  }

  return;
}

/****************************************************************************
 *
 *  init_boundary_speeds
 *
 ****************************************************************************/

static void init_boundary_speeds(const double ux_bot, const double ux_top) {

  B_link * p_link;

  p_link = link_list_;

  while (p_link) {

    /* Decide whether the link is at the top or bottom */

    if (cv[p_link->p][Z] == -1) {
      p_link->ux = ux_bot;
    }
    if (cv[p_link->p][Z] == +1) {
      p_link->ux = ux_top;
    }

    p_link = p_link->next;
  }

  return;
}

/****************************************************************************
 *
 *  allocate_link
 *
 *  Return a pointer to a newly allocated boundary link structure
 *  or fail gracefully.
 *
 ****************************************************************************/

B_link * allocate_link() {

  B_link * p_link;

  p_link = (B_link *) malloc(sizeof(B_link));

  if (p_link == (B_link *) NULL) {
    fatal("malloc(B_link) failed\n");
  }

  nalloc_links_++;

  return p_link;
}

/*****************************************************************************
 *
 *  set_wall_velocity
 *
 *  Set distribution at solid sites to reflect solid body velocity.
 *  This allows 'solid-solid' exchange of distributions between
 *  wall and colloids.
 *
 *****************************************************************************/

static void set_wall_velocity() {

  B_link * p_link;
  double   fp;
  double   rho;
  int      p;

  rho = get_rho0();
  p_link = link_list_;

  while (p_link) {
    p = NVEL - p_link->p; /* Want the outward going component */
    fp = wv[p]*(rho + rcs2*p_link->ux*cv[p][X]);
    distribution_f_set(p_link->j, p, 0, fp);

    p_link = p_link->next;
  }

  return;
}

/*****************************************************************************
 *
 *  wall_accumulate_force
 *
 *  Add a contribution to the force on the walls. This is for accounting
 *  purposes only. There is no physical consequence.
 *
 *****************************************************************************/

void wall_accumulate_force(const double f[3]) {

  int ia;

  for (ia = 0; ia < 3; ia++) {
    fnet_[ia] += f[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  wall_net_momentum
 *
 *  Get the accumulated force (interpreted as momentum) on the walls.
 *
 *  This is a reduction to rank 0 in pe_comm() for the purposes
 *  of output statistics. This is the only meaningful use of this
 *  quantity.
 *
 *****************************************************************************/

void wall_net_momentum(double g[3]) {

  MPI_Reduce(fnet_, g, 3, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

  return;
}

/*****************************************************************************
 *
 *  wall_shear_init
 *
 *  Initialise the distributions to be consistent with a linear shear
 *  profile for the given top and bottom wall velocities.
 *
 *  This is only relevant for walls at z = 0 and z = L_z.
 *
 *****************************************************************************/

static void wall_shear_init(double uxtop, double uxbottom) {

  int ic, jc, kc, index;
  int ia, ib, p;
  int nlocal[3];
  int noffset[3];
  double rho, u[3], gradu[3][3];
  double eta;
  double gammadot;
  double f;
  double cdotu;
  double sdotq;

  /* Shear rate */
  gammadot = (uxtop - uxbottom)/L(Z);

  info("Initialising linear shear profile for walls\n");
  info("Speed at top u_x    %14.7e\n", uxtop);
  info("Speed at bottom u_x %14.7e\n", uxbottom); 
  info("Overall shear rate  %14.7e\n", gammadot);

  /* Initialise the density, velocity, gradu; ghost modes are zero */

  rho = get_rho0();
  eta = get_eta_shear();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ia = 0; ia < 3; ia++) {
    u[ia] = 0.0;
    for (ib = 0; ib < 3; ib++) {
      gradu[ia][ib] = 0.0;
    }
  }

  /* Shear rate */
  gradu[X][Z] = gammadot;


  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	/* Linearly interpolate between top and bottom to get velocity;
	 * the - 1.0 accounts for kc starting at 1. */
	u[X] = uxbottom + (noffset[Z] + kc - 0.5)*(uxtop - uxbottom)/L(Z);

        index = coords_index(ic, jc, kc);

        for (p = 0; p < NVEL; p++) {

	  cdotu = 0.0;
	  sdotq = 0.0;

          for (ia = 0; ia < 3; ia++) {
            cdotu += cv[p][ia]*u[ia];
            for (ib = 0; ib < 3; ib++) {
              sdotq += (rho*u[ia]*u[ib] - eta*gradu[ia][ib])*q_[p][ia][ib];
            }
          }
          f = wv[p]*rho*(1.0 + rcs2*cdotu + 0.5*rcs2*rcs2*sdotq);
          distribution_f_set(index, p, 0, f);
        }
        /* Next site */
      }
    }
  }

  return;
}
