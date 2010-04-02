/*****************************************************************************
 *
 *  wall.c
 *
 *  Static solid objects.
 *
 *  Special case: boundary walls.
 *
 *  $Id: wall.c,v 1.11.4.5 2010-04-02 07:56:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
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
static void     report_boundary_memory(void);
static void     set_wall_velocity(void);

/*****************************************************************************
 *
 *  wall_init
 *
 ****************************************************************************/

void wall_init() {

  double ux_bottom = 0.0;
  double ux_top = 0.0;
  char   tmp[128];

  RUN_get_double_parameter("boundary_speed_bottom", &ux_bottom);
  RUN_get_double_parameter("boundary_speed_top", &ux_top);
  RUN_get_string_parameter("boundary_walls_on", tmp, 128);

  if (strcmp(tmp, "yes") == 0) is_boundary_wall_ = 1;

  if (is_boundary_wall_) init_boundary_site_map();
  init_links();
  if (is_boundary_wall_) init_boundary_speeds(ux_bottom, ux_top);

  report_boundary_memory();

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

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 0; ic <= nlocal[X] + 1; ic++) {
    for (jc = 0; jc <= nlocal[Y] + 1; jc++) {
      for (kc = 0; kc <= nlocal[Z] + 1; kc++) {

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
 *  report_boundary_memory
 *
 *****************************************************************************/

static void report_boundary_memory() {

  info("[Boundary links: %d (%d bytes)]\n", nalloc_links_,
       nalloc_links_*sizeof(B_link));

  return;
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
 *  This is a reduction to rank 0 in MPI_COMM_WORLD for the purposes
 *  of output statistics. This is the only meaningful use of this
 *  quantity.
 *
 *****************************************************************************/

void wall_net_momentum(double g[3]) {

  MPI_Reduce(fnet_, g, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return;
}
