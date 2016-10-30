/*****************************************************************************
 *
 *  wall.c
 *
 *  Static solid objects (porous media).
 *
 *  Special case: boundary walls. The two issues might be sepatated.
 *
 *  Edinburgh Soft Matter and Statistical Physics and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2016 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "physics.h"
#include "util.h"
#include "wall.h"

typedef enum wall_init_enum {WALL_INIT_COUNT_ONLY,
			     WALL_INIT_ALLOCATE} wall_init_enum_t;

typedef enum wall_uw_enum {WALL_UZERO = 0,
			   WALL_UWTOP,
			   WALL_UWBOT,
			   WALL_UWMAX} wall_uw_enum_t;

struct wall_s {
  pe_t * pe;             /* Parallel environment */
  cs_t * cs;             /* Reference to coordinate system */
  map_t * map;           /* Reference to map structure */
  lb_t * lb;             /* Reference to LB information */ 
  wall_t * target;       /* Device memory */

  wall_param_t * param;  /* parameters */
  int   nlink;           /* Number of links */
  int * linki;           /* outside (fluid) site indices */
  int * linkj;           /* inside (solid) site indices */
  int * linkp;           /* LB basis vectors for links */
  int * linku;           /* Link wall_uw_enum_t (wall velocity) */
  double fnet[3];        /* Momentum accounting for source/sink walls */
};

int wall_init_boundaries(wall_t * wall, wall_init_enum_t init);
int wall_init_map(wall_t * wall);
int wall_init_uw(wall_t * wall);

static __constant__ wall_param_t static_param;

/*****************************************************************************
 *
 *  wall_create
 *
 *****************************************************************************/

__host__ int wall_create(pe_t * pe, cs_t * cs, map_t * map, lb_t * lb,
			 wall_t ** p) {

  int ndevice;
  wall_t * wall = NULL;

  assert(pe);
  assert(cs);
  assert(p);

  wall = (wall_t *) calloc(1, sizeof(wall_t));
  if (wall == NULL) pe_fatal(pe, "calloc(wall_t) failed\n");

  wall->param = (wall_param_t *) calloc(1, sizeof(wall_param_t));
  if (wall->param == NULL) pe_fatal(pe, "calloc(wall_param_t) failed\n");

  wall->pe = pe;
  wall->cs = cs;
  wall->map = map;
  wall->lb = lb;

  cs_retain(cs);

  /* Target copy */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    wall->target = wall;
  }
  else {
    wall_param_t * tmp = NULL;

    targetCalloc((void **) &wall->target, sizeof(wall_t));
    targetConstAddress((void **) &tmp, static_param);
    copyToTarget(&wall->target->param, &tmp, sizeof(wall_param_t *));
  }

  *p = wall;

  return 0;
}

/*****************************************************************************
 *
 *  wall_free
 *
 *****************************************************************************/

__host__ int wall_free(wall_t * wall) {

  assert(wall);

  if (wall->target != wall) {
    targetFree(wall->target);
  }

  cs_free(wall->cs);
  free(wall->param);
  if (wall->linki) free(wall->linki);
  if (wall->linkj) free(wall->linkj);
  if (wall->linkp) free(wall->linkp);
  if (wall->linku) free(wall->linku);
  free(wall);

  return 0;
}

/*****************************************************************************
 *
 *  wall_commit
 *
 *****************************************************************************/

__host__ int wall_commit(wall_t * wall, wall_param_t param) {

  assert(wall);

  *wall->param = param;

  wall_init_map(wall);
  wall_init_boundaries(wall, WALL_INIT_COUNT_ONLY);
  wall_init_boundaries(wall, WALL_INIT_ALLOCATE);
  wall_init_uw(wall);

  return 0;
}

/*****************************************************************************
 *
 *  wall_info
 *
 *  Note a global communication.
 *
 *****************************************************************************/

__host__ int wall_info(wall_t * wall) {

  int nlink;
  pe_t * pe = NULL;
  MPI_Comm comm;

  assert(wall);

  pe = wall->pe;

  pe_mpi_comm(pe, &comm);
  MPI_Reduce(&wall->nlink, &nlink, 1, MPI_INT, MPI_SUM, 0, comm);

  if (wall->param->iswall) {
    pe_info(pe, "\n");
    pe_info(pe, "Boundary walls\n");
    pe_info(pe, "--------------\n");
    pe_info(pe, "Boundary walls:                  %1s %1s %1s\n",
	    (wall->param->isboundary[X] == 1) ? "X" : "-",
	    (wall->param->isboundary[Y] == 1) ? "Y" : "-",
	    (wall->param->isboundary[Z] == 1) ? "Z" : "-");
    pe_info(pe, "Boundary speed u_x (bottom):    %14.7e\n",
	    wall->param->ubot[X]);
    pe_info(pe, "Boundary speed u_x (top):       %14.7e\n",
	    wall->param->utop[X]);
    pe_info(pe, "Boundary normal lubrication rc: %14.7e\n",
	    wall->param->lubr_rc[X]);

    pe_info(pe, "Wall boundary links allocated:   %d\n", nlink);
    pe_info(pe, "Memory (total, bytes):           %d\n", 4*nlink*sizeof(int));
    pe_info(pe, "Boundary shear initialise:       %d\n",
	    wall->param->initshear);
  }

  if (wall->param->isporousmedia) {
    pe_info(pe, "\n");
    pe_info(pe, "Porous Media\n");
    pe_info(pe, "------------\n");
    pe_info(pe, "Wall boundary links allocated:   %d\n", nlink);
    pe_info(pe, "Memory (total, bytes):           %d\n", 4*nlink*sizeof(int));
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_target
 *
 *****************************************************************************/

__host__ int wall_target(wall_t * wall, wall_t ** target) {

  assert(wall);
  assert(target);

  *target = wall->target;

  return 0;
}

/*****************************************************************************
 *
 *  wall_param_set
 *
 *****************************************************************************/

__host__ int wall_param_set(wall_t * wall, wall_param_t values) {

  assert(wall);

  *wall->param = values;

  return 0;
}

/*****************************************************************************
 *
 *  wall_param
 *
 *****************************************************************************/

__host__ int wall_param(wall_t * wall, wall_param_t * values) {

  assert(wall);
  assert(values);

  *values = *wall->param;

  return 0;
}

/*****************************************************************************
 *
 *  wall_init_boundaries
 *
 *****************************************************************************/

__host__ int wall_init_boundaries(wall_t * wall, wall_init_enum_t init) {

  int ic, jc, kc;
  int ic1, jc1, kc1;
  int indexi, indexj;
  int p;
  int nlink;
  int nlocal[3];
  int status;

  assert(wall);

  if (init == WALL_INIT_ALLOCATE) {
    wall->linki = (int *) calloc(wall->nlink, sizeof(int));
    wall->linkj = (int *) calloc(wall->nlink, sizeof(int));
    wall->linkp = (int *) calloc(wall->nlink, sizeof(int));
    wall->linku = (int *) calloc(wall->nlink, sizeof(int));
    if (wall->linki == NULL) pe_fatal(wall->pe,"calloc(wall->linki) failed\n");
    if (wall->linkj == NULL) pe_fatal(wall->pe,"calloc(wall->linkj) failed\n");
    if (wall->linkp == NULL) pe_fatal(wall->pe,"calloc(wall->linkp) failed\n");
    if (wall->linku == NULL) pe_fatal(wall->pe,"calloc(wall->linku) failed\n");
  }

  nlink = 0;
  cs_nlocal(wall->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	indexi = cs_index(wall->cs, ic, jc, kc);
	map_status(wall->map, indexi, &status);
	if (status != MAP_FLUID) continue;

	/* Look for non-solid -> solid links */

	for (p = 1; p < NVEL; p++) {

	  ic1 = ic + cv[p][X];
	  jc1 = jc + cv[p][Y];
	  kc1 = kc + cv[p][Z];
	  indexj = cs_index(wall->cs, ic1, jc1, kc1);
	  map_status(wall->map, indexj, &status);

	  if (status == MAP_BOUNDARY) {
	    if (init == WALL_INIT_ALLOCATE) {
	      wall->linki[nlink] = indexi;
	      wall->linkj[nlink] = indexj;
	      wall->linkp[nlink] = p;
	      wall->linku[nlink] = WALL_UZERO;
	    }
	    nlink += 1;
	  }
	}

	/* Next site */
      }
    }
  }

  if (init) {
    assert(nlink == wall->nlink);
  }
  wall->nlink = nlink;

  return 0;
}

/*****************************************************************************
 *
 *  wall_init_uw
 *
 *  Only the simple case of one set of walls is handled at present.
 *
 *****************************************************************************/

__host__ int wall_init_uw(wall_t * wall) {

  int n;
  int iw;
  int nwall;

  assert(wall);

  nwall = wall->param->isboundary[X] + wall->param->isboundary[Y]
    + wall->param->isboundary[Z];

  if (nwall == 1) {
    /* All links are either top or bottom */
    if (wall->param->isboundary[X]) iw = X;
    if (wall->param->isboundary[Y]) iw = Y;
    if (wall->param->isboundary[Z]) iw = Z;

    for (n = 0; n < wall->nlink; n++) {
      if (cv[wall->linkp[n]][iw] == -1) wall->linku[n] = WALL_UWBOT;
      if (cv[wall->linkp[n]][iw] == +1) wall->linku[n] = WALL_UWTOP;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_set_wall_distribution
 *
 *  Set distribution at solid sites to reflect the solid body velocity.
 *  This allows 'solid-solid' exchange of distributions between wall
 *  and colloids.
 *
 *****************************************************************************/

__host__ int wall_set_wall_distributions(wall_t * wall) {

  int n;
  int p;              /* Outward going component of link velocity */
  double rho, fp;
  double ux = 0.0;    /* PENDING initialisation */
  physics_t * phys = NULL;

  assert(wall);
  physics_ref(&phys);
  physics_rho0(phys, &rho);

  for (n = 0; n < wall->nlink; n++) {
    p = NVEL - wall->linkp[n];
    fp = wv[p]*(rho + rcs2*ux*cv[p][X]);
    lb_f_set(wall->lb, wall->linkj[n], p, LB_RHO, fp);
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_bbl
 *
 *****************************************************************************/

__host__ int wall_bbl(wall_t * wall) {

  int i, j, ij, ji, ia;
  int n, ndist;
  int status;

  double uw[WALL_UWMAX][3];
  double rho, cdotu;
  double fp, fp0, fp1;
  double force;

  assert(wall);

  /* Load the current wall velocities into the uw table */

  for (ia = 0; ia < 3; ia++) {
    uw[WALL_UZERO][ia] = 0.0;
    uw[WALL_UWTOP][ia] = wall->param->utop[ia];
    uw[WALL_UWBOT][ia] = wall->param->ubot[ia];
  }

  lb_ndist(wall->lb, &ndist);

  for (n = 0; n < wall->nlink; n++) {

    i  = wall->linki[n];
    j  = wall->linkj[n];
    ij = wall->linkp[n];   /* Link index direction solid->fluid */
    ji = NVEL - ij;        /* Opposite direction index */
    ia = wall->linku[n];   /* Wall velocity lookup */

    cdotu = cv[ij][X]*uw[ia][X] + cv[ij][Y]*uw[ia][Y] + cv[ij][Z]*uw[ia][Z]; 

    map_status(wall->map, i, &status);

    if (status == MAP_COLLOID) {

      /* This matches the momentum exchange in colloid BBL. */
      /* This only affects the accounting (via anomaly, as below) */

      lb_f(wall->lb, i, ij, LB_RHO, &fp0);
      lb_f(wall->lb, j, ji, LB_RHO, &fp1);
      fp = fp0 + fp1;
      for (ia = 0; ia < 3; ia++) {
	wall->fnet[ia] += (fp - 2.0*wv[ij])*cv[ij][ia];
      }

    }
    else {

      /* This is the momentum. To prevent accumulation of round-off
       * in the running total (fnet_), we subtract the equilibrium
       * wv[ij]. This is ok for walls where there are exactly
       * equal and opposite links at each side of the system. */

      lb_f(wall->lb, i, ij, LB_RHO, &fp);
      lb_0th_moment(wall->lb, i, LB_RHO, &rho);

      force = 2.0*fp - 2.0*rcs2*wv[ij]*rho*cdotu;
      for (ia = 0; ia < 3; ia++) {
	wall->fnet[ia] += (force - 2.0*wv[ij])*cv[ij][ia];
      }

      fp = fp - 2.0*rcs2*wv[ij]*rho*cdotu;
      lb_f_set(wall->lb, j, ji, LB_RHO, fp);

      if (ndist > 1) {
	/* Order parameter */
	lb_f(wall->lb, i, ij, LB_PHI, &fp);
	lb_0th_moment(wall->lb, i, LB_PHI, &rho);

	fp = fp - 2.0*rcs2*wv[ij]*rho*cdotu;
	lb_f_set(wall->lb, j, ji, LB_PHI, fp);
      }

    }
    /* Next link */
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_init_map
 *
 *****************************************************************************/

__host__ int wall_init_map(wall_t * wall) {

  int ic, jc, kc, index;
  int ic_global, jc_global, kc_global;
  int ntotal[3];
  int nlocal[3];
  int noffset[3];
  int nextra;

  assert(wall);

  cs_ntotal(wall->cs, ntotal);
  cs_nlocal(wall->cs, nlocal);
  cs_nlocal_offset(wall->cs, noffset);
  cs_nhalo(wall->cs, &nextra);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	/* If this is an appropriate periodic boundary, set to solid */

	ic_global = ic + noffset[X];
	jc_global = jc + noffset[Y];
	kc_global = kc + noffset[Z];

	if (wall->param->isboundary[Z]) {
	  if (kc_global == 0 || kc_global == ntotal[Z] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}

	if (wall->param->isboundary[Y]) {
	  if (jc_global == 0 || jc_global == ntotal[Y] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}

	if (wall->param->isboundary[X]) {
	  if (ic_global == 0 || ic_global == ntotal[X] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}
	/* next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_momentum_add
 *
 *****************************************************************************/

__host__ __device__ int wall_momentum_add(wall_t * wall, const double f[3]) {

  assert(wall);

  wall->fnet[X] += f[X];
  wall->fnet[Y] += f[Y];
  wall->fnet[Z] += f[Z];

  return 0;
}

/*****************************************************************************
 *
 *  wall_momentum
 *
 *****************************************************************************/

__host__ __device__ int wall_momentum(wall_t * wall, double f[3]) {

  assert(wall);

  f[X] = wall->fnet[X];
  f[Y] = wall->fnet[Y];
  f[Z] = wall->fnet[Z];

  return 0;
}

/*****************************************************************************
 *
 *  wall_is_pm
 *
 *****************************************************************************/

__host__ __device__ int wall_is_pm(wall_t * wall, int * ispm) {

  assert(wall);

  *ispm = wall->param->isporousmedia;

  return 0;
}

/*****************************************************************************
 *
 *  wall_present
 *
 *****************************************************************************/

__host__ __device__ int wall_present(wall_t * wall) {

  wall_param_t * wp = NULL;

  assert(wall);

  wp = wall->param;
  return (wp->isboundary[X] || wp->isboundary[Y] || wp->isboundary[Z]);
}

/*****************************************************************************
 *
 *  wall_present_dim
 *
 *****************************************************************************/

__host__ __device__ int wall_present_dim(wall_t * wall, int iswall[3]) {

  assert(wall);

  iswall[X] = wall->param->isboundary[X];
  iswall[Y] = wall->param->isboundary[Y];
  iswall[Z] = wall->param->isboundary[Z];

  return 0;
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

__host__ int wall_shear_init(wall_t * wall) {

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
  double uxbottom;
  double uxtop;
  double ltot[3];
  physics_t * phys = NULL;

  assert(wall);

  /* One wall constraint */
  uxtop = wall->param->utop[X];
  uxbottom = wall->param->ubot[X];

  cs_ltot(wall->cs, ltot);

  /* Shear rate */
  gammadot = (uxtop - uxbottom)/ltot[Z];

  pe_info(wall->pe, "Initialising linear shear profile for walls\n");
  pe_info(wall->pe, "Speed at top u_x    %14.7e\n", uxtop);
  pe_info(wall->pe, "Speed at bottom u_x %14.7e\n", uxbottom); 
  pe_info(wall->pe, "Overall shear rate  %14.7e\n", gammadot);

  /* Initialise the density, velocity, gradu; ghost modes are zero */

  physics_ref(&phys);
  physics_rho0(phys, &rho);
  physics_eta_shear(phys, &eta);

  cs_nlocal(wall->cs, nlocal);
  cs_nlocal_offset(wall->cs, noffset);

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
	u[X] = uxbottom + (noffset[Z] + kc - 0.5)*(uxtop - uxbottom)/ltot[Z];

        index = cs_index(wall->cs, ic, jc, kc);

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
          lb_f_set(wall->lb, index, p, 0, f);
        }
        /* Next site */
      }
    }
  }

  return 0;
}

/******************************************************************************
 *
 *  wall_lubrication
 *
 *  This returns the normal lubrication correction for colloid of hydrodynamic
 *  radius ah at position r near a flat wall in dimension dim (if present).
 *  This is based on the analytical expression for a sphere.
 *
 *  The result should be added to the appropriate diagonal element of
 *  the colloid's drag matrix in the implicit update. There is, therefore,
 *  no velocity appearing here (wall assumed to have no velocity).
 *  This is therefore closely related to BBL in bbl.c.
 *
 *  This operates in parallel by computing the absolute distance between
 *  the side of the system (walls nominally at Lmin and (Lmax + Lmin)),
 *  and applying the cutoff.
 * 
 *  Normal force is added to the diagonal of drag matrix \zeta^FU_xx etc
 *  (No tangential force would be added to \zeta^FU_xx and \zeta^FU_yy)
 *
 *****************************************************************************/

__host__ int wall_lubr_sphere(wall_t * wall, double ah, const double r[3],
			      double * drag) {

  double hlub;
  double h;
  double eta;
  double lmin[3];
  double ltot[3];
  physics_t * phys = NULL;
  PI_DOUBLE(pi);

  drag[X] = 0.0;
  drag[Y] = 0.0;
  drag[Z] = 0.0;

  if (wall == NULL) return 0; /* PENDING prefer assert()? */

  cs_lmin(wall->cs, lmin);
  cs_ltot(wall->cs, ltot);

  physics_ref(&phys);
  physics_eta_shear(phys, &eta);

  /* Lower, then upper wall X, Y, and Z */

  if (wall->param->isboundary[X]) {
    hlub = wall->param->lubr_rc[X];
    h = r[X] - lmin[X] - ah; 
    if (h < hlub) drag[X] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
    h = lmin[X] + ltot[X] - r[X] - ah;
    if (h < hlub) drag[X] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
  }

  if (wall->param->isboundary[Y]) {
    hlub = wall->param->lubr_rc[Y];
    h = r[Y] - lmin[Y] - ah; 
    if (h < hlub) drag[Y] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
    h = lmin[Y] + ltot[Y] - r[Y] - ah;
    if (h < hlub) drag[Y] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
  }

  if (wall->param->isboundary[Z]) {
    hlub = wall->param->lubr_rc[Z];
    h = r[Z] - lmin[Z] - ah; 
    if (h < hlub) drag[Z] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
    h = lmin[Z] + ltot[Z] - r[Z] - ah;
    if (h < hlub) drag[Z] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
  }

  return 0;
}
