/*****************************************************************************
 *
 *  build_links.c
 *
 *  Construct the set of links which make up a BBL particle.
 *
 *  There are currently two implementations:
 *    1. host version to use linked list
 *    2. host/device version to use the link array.
 *
 *  The linked list version will be removed at some point in the future.
 *
 *
 *  (c) 2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "build_links.h"

/* FIXME shift these routines please */

__host__ __device__ static inline int util_imax(int a, int b) {

  return (a > b) ? a : b;
}

__host__ __device__ static inline int util_imin(int a, int b) {

  return (a < b) ? a : b;
}

__host__ __device__ static inline int
util_square_modulus_int8(const int8_t cv[3]) {

  return (int) (cv[0] * cv[0] + cv[1] * cv[1] + cv[2] * cv[2]);
}

__host__ __device__ static inline int cs_index_to_ic(const cs_t * cs,
                                                     int          index) {

  assert(cs);

  return ((1 - cs->param->nhalo) + index / cs->param->str[X]);
}

__host__ __device__ static inline int cs_index_to_jc(const cs_t * cs,
                                                     int          index) {

  int jc = 0;
  assert(cs);

  jc =
      (1 - cs->param->nhalo) + (index % cs->param->str[X]) / cs->param->str[Y];

  return jc;
}

__host__ __device__ static inline int cs_index_to_kc(const cs_t * cs,
                                                     int          index) {

  int kc = 0;

  assert(cs);

  kc = (1 - cs->param->nhalo) + index % cs->param->str[Y];

  return kc;
}

/****************************************************************************
 *
 *  build_links_colloid_fluid
 *
 ****************************************************************************/

int build_links_colloid_fluid(colloids_info_t * info, map_t * map,
                              const lb_model_t * model, colloid_t * pc) {

  const double lambda = 0.5;

  int i_min, i_max;
  int j_min, j_max;
  int k_min, k_max;
  int nlocal[3] = {};

  double amax = 0.0;

  assert(model);
  assert(pc);

  colloid_link_t * link = NULL;

  /* Make sure we have a head */

  if (pc->lnk == NULL) {
    pc->lnk = colloid_link_allocate();
  }

  /* Unset all links */

  link = pc->lnk;

  while (link) {
    link->status = LINK_UNUSED;
    link         = link->next;
  }

  /* Limits of the search region around the particle. This has to be large
   * enough to capture any local links. Links are "outside to inside" and
   * the outside is not in the halo region. (Such links are captured by the
   * image particle in a neighbouring process.) */

  cs_nlocal(info->cs, nlocal);

  amax = colloid_principal_radius(&pc->s);

  {
    /* Local limits require colloid position minus offset */
    double rc[3] = {};

    rc[X] = pc->s.r[X] - 1.0 * map->cs->param->noffset[X];
    rc[Y] = pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y];
    rc[Z] = pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z];

    i_min = util_imax(1, (int) floor(rc[X] - amax));
    j_min = util_imax(1, (int) floor(rc[Y] - amax));
    k_min = util_imax(1, (int) floor(rc[Z] - amax));
    i_max = util_imin(nlocal[X], (int) ceil(rc[X] + amax));
    j_max = util_imin(nlocal[Y], (int) ceil(rc[Y] + amax));
    k_max = util_imin(nlocal[Z], (int) ceil(rc[Z] + amax));
  }

  /* Begin search ... */

  link = pc->lnk;

  for (int ic = i_min; ic <= i_max; ic++) {
    for (int jc = j_min; jc <= j_max; jc++) {
      for (int kc = k_min; kc <= k_max; kc++) {

        /* We are looking for links i -> j where i is outside and j is
         * inside; i is a local site (not halo) */

        int    indexi = cs_index(info->cs, ic, jc, kc);
        int    status = MAP_FLUID;
        double r0[3]  = {1.0 * ic, 1.0 * jc, 1.0 * kc};

        colloid_t * pchere = NULL;

        colloids_info_map(info, indexi, &pchere);
        if (pchere == pc) {
          continue;
        }

        map_status(map, indexi, &status);

        /* Index i is outside, so cycle through the lattice vectors
         * to determine if the end is inside, and so requires a link */

        for (int p = 1; p < model->nvel; p++) {

          /* Find the index of the potential inside site */

          int ii = ic + model->cv[p][X];
          int jj = jc + model->cv[p][Y];
          int kk = kc + model->cv[p][Z];

          int    indexj = cs_index(info->cs, ii, jj, kk);
          double rb[3]  = {}; /* centre -> local site (i, j, k) */

          colloids_info_map(info, indexj, &pchere);
          if (pchere != pc) {
            continue;
          }

          /* Index j is inside, so initialise the link */

          rb[X] = r0[X] - (pc->s.r[X] - map->cs->param->noffset[X]);
          rb[Y] = r0[Y] - (pc->s.r[Y] - map->cs->param->noffset[Y]);
          rb[Z] = r0[Z] - (pc->s.r[Z] - map->cs->param->noffset[Z]);

          link->rb[X] = rb[X] + lambda * model->cv[p][X];
          link->rb[Y] = rb[Y] + lambda * model->cv[p][Y];
          link->rb[Z] = rb[Z] + lambda * model->cv[p][Z];

          link->i      = indexi;
          link->j      = indexj;
          link->p      = p;
          link->status = LINK_COLLOID;

          if (status == MAP_FLUID) {
            link->status = LINK_FLUID;
          }

          /* Add a new unused link here in case we need it next time */
          if (link->next == NULL) {
            link->next = colloid_link_allocate();
          }

          link = link->next;
        }

        /* Next site in the search */
      }
    }
  }

  return 0;
}

/****************************************************************************
 *
 *  build_links_colloid_wall
 *
 *  There's scope for some rationalisation between this routine and the
 *  corresponding fluid one above.
 *
 *  The difference is that for the boundary link case, links are allowed
 *  to start in the halo region, where the plane wall are located. Flkuid
 *  links cannot start in the halo region.
 *
 *  However, much is very similar.
 *
 ****************************************************************************/

int build_links_colloid_wall(colloids_info_t * info, map_t * map,
                             wall_t * wall, const lb_model_t * model,
                             colloid_t * pc) {

  const double lambda = 0.5;

  int i_min, i_max;
  int j_min, j_max;
  int k_min, k_max;
  int nlocal[3] = {};

  double amax = 0.0;

  assert(model);
  assert(pc);
  assert(pc->lnk);

  colloid_link_t * link = NULL;

  /* Limits of the search region around the particle. This has to be large
   * enough to capture any local links. Links are "outside to inside" and
   * the outside is not in the halo region. (Such links are captured by the
   * image particle in a neighbouring process.) */

  cs_nlocal(info->cs, nlocal);

  amax = colloid_principal_radius(&pc->s);

  {
    /* Local limits require colloid position minus offset */
    double rc[3] = {};

    rc[X] = pc->s.r[X] - 1.0 * map->cs->param->noffset[X];
    rc[Y] = pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y];
    rc[Z] = pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z];

    i_min = util_imax(1, (int) floor(rc[X] - amax));
    j_min = util_imax(1, (int) floor(rc[Y] - amax));
    k_min = util_imax(1, (int) floor(rc[Z] - amax));
    i_max = util_imin(nlocal[X], (int) ceil(rc[X] + amax));
    j_max = util_imin(nlocal[Y], (int) ceil(rc[Y] + amax));
    k_max = util_imin(nlocal[Z], (int) ceil(rc[Z] + amax));
  }

  /* Begin search ... */

  link = pc->lnk;
  while (link && link->status != LINK_UNUSED) {
    link = link->next;
  }

  for (int ic = i_min; ic <= i_max; ic++) {
    int inear = (ic == 1 || ic == nlocal[X]) * wall->param->isboundary[X];
    for (int jc = j_min; jc <= j_max; jc++) {
      int jnear = (jc == 1 || jc == nlocal[Y]) * wall->param->isboundary[Y];
      for (int kc = k_min; kc <= k_max; kc++) {
        int knear = (kc == 1 || kc == nlocal[Z]) * wall->param->isboundary[Z];

        /* We are looking for links i -> j where i is outside and j is
         * inside; i (outside) must be a BOUNDARY */

        int    indexj = cs_index(info->cs, ic, jc, kc);
        double r0[3]  = {1.0 * ic, 1.0 * jc, 1.0 * kc};

        colloid_t * pchere = NULL;

        /* We need to be near the perimeter */
        /* We need indexj inside colloid */
        if (0 == (inear || jnear || knear)) {
          continue;
        }

        colloids_info_map(info, indexj, &pchere);
        if (pchere != pc) {
          continue;
        }

        /* The search is now "inside out" ... */

        for (int p = 1; p < model->nvel; p++) {

          /* Find the index of the potential outside site */

          int ii = ic + model->cv[p][X];
          int jj = jc + model->cv[p][Y];
          int kk = kc + model->cv[p][Z];

          int    indexi = cs_index(info->cs, ii, jj, kk);
          int    status = MAP_FLUID;
          double rb[3]  = {}; /* centre -> local site (i, j, k) */

          map_status(map, indexi, &status);
          if (status != MAP_BOUNDARY) {
            continue;
          }

          /* Index i is boundary, so initialise the link */

          rb[X] = r0[X] - (pc->s.r[X] - map->cs->param->noffset[X]);
          rb[Y] = r0[Y] - (pc->s.r[Y] - map->cs->param->noffset[Y]);
          rb[Z] = r0[Z] - (pc->s.r[Z] - map->cs->param->noffset[Z]);

          link->rb[X] = rb[X] + lambda * model->cv[p][X];
          link->rb[Y] = rb[Y] + lambda * model->cv[p][Y];
          link->rb[Z] = rb[Z] + lambda * model->cv[p][Z];

          link->i      = indexi;
          link->j      = indexj;
          link->p      = model->nvel - p; /* Opposite of search direction */
          link->status = LINK_BOUNDARY;

          /* Add a new unused link here in case we need it next time */
          if (link->next == NULL) {
            link->next = colloid_link_allocate();
          }

          link = link->next;
        }

        /* Next site in the search */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_reset_links
 *
 *  Recompute the boundary link vectors and solid/fluid status
 *  of links for an existing particle.
 *
 ****************************************************************************/

__host__ __device__ void build_links_reset_colloid(colloid_t *        pc,
                                                   const lb_model_t * model,
                                                   map_t *            map) {

  const double lambda = 0.5;

  assert(pc);
  assert(map);
  assert(model);

  for (colloid_link_t * link = pc->lnk; link; link = link->next) {

    if (link->status == LINK_UNUSED) {
      continue;
    }

    /* Compute the separation between the centre of the colloid
     * and the fluid site involved with this link. The position
     * of the outside site is rsite in local coordinates. */

    int ic = cs_index_to_ic(map->cs, link->i);
    int jc = cs_index_to_jc(map->cs, link->i);
    int kc = cs_index_to_kc(map->cs, link->i);

    int status = MAP_COLLOID;

    double r0[3] = {1.0 * ic, 1.0 * jc, 1.0 * kc}; /* this site (local) */
    double rb[3] = {};                             /* colloid centre -> site */

    rb[X] = r0[X] - (pc->s.r[X] - 1.0 * map->cs->param->noffset[X]);
    rb[Y] = r0[Y] - (pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y]);
    rb[Z] = r0[Z] - (pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z]);

    link->rb[X] = rb[X] + lambda * model->cv[link->p][X];
    link->rb[Y] = rb[Y] + lambda * model->cv[link->p][Y];
    link->rb[Z] = rb[Z] + lambda * model->cv[link->p][Z];

    map_status(map, link->i, &status);

    if (status == MAP_FLUID) {
      link->status = LINK_FLUID;
    }
    else {
      /* Could try to avoid this translation (e.g., make LINK_X = MAP_X) ... */
      if (status == MAP_COLLOID) {
        link->status = LINK_COLLOID;
      }
      if (status == MAP_BOUNDARY) {
        link->status = LINK_BOUNDARY;
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_links_evaluate_mean
 *
 *  Evaluate sum of weights, and sum of two vector quantities.
 *
 *  Links are a serial loop in the device implementation.
 *
 *****************************************************************************/

__host__ __device__ int build_links_evaluate_mean(colloid_t *        pc,
                                                  const lb_model_t * model) {

  /* Evaluate sum of link weights */
  /* Evaluate cbar[] and rxcbar[] */

  pc->sumw = 0.0;

  for (int ia = 0; ia < 3; ia++) {
    pc->cbar[ia]   = 0.0;
    pc->rxcbar[ia] = 0.0;
  }

  /* Only fluid links count ... */

  for (colloid_link_t * link = pc->lnk; link; link = link->next) {

    if (link->status == LINK_FLUID) {

      double wv      = model->wv[link->p];
      double wvc[3]  = {};
      double rbxc[3] = {};

      pc->sumw += wv;

      wvc[X] = wv * model->cv[link->p][X];
      wvc[Y] = wv * model->cv[link->p][Y];
      wvc[Z] = wv * model->cv[link->p][Z];

      util_vector_cross_product(rbxc, link->rb, wvc);

      for (int ia = 0; ia < 3; ia++) {
        pc->cbar[ia] += wvc[ia];
        pc->rxcbar[ia] += rbxc[ia];
      }
    }

    /* Next link */
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_links_evaluate_area
 *
 *  Count number of faces (local) for this colloid. This is the 'surface
 *  area' on the finite difference grid.
 *
 *  Count both total, and those faces which have fluid neighbours.
 *
 *  Must be serialised on device at the moment.
 *
 *****************************************************************************/

__host__ __device__ int build_links_evaluate_area(colloid_t *        pc,
                                                  const lb_model_t * model) {

  assert(pc);
  assert(model);

  pc->s.sa  = 0.0;
  pc->s.saf = 0.0;

  for (colloid_link_t * link = pc->lnk; link; link = link->next) {
    if (link->status == LINK_UNUSED) {
      continue;
    }
    int p = util_square_modulus_int8(model->cv[link->p]);
    if (p == 1) {
      pc->s.sa += 1.0;
      if (link->status == LINK_FLUID) {
        pc->s.saf += 1.0;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_links_update_links_colloid
 *
 *  Update the links for a single colloid.
 *
 *****************************************************************************/

int build_links_update_links_colloid(colloids_info_t *  info,
                                     const lb_model_t * model, map_t * map,
                                     wall_t * wall, colloid_t * pc) {
  assert(pc);
  assert(wall);

  if (pc->s.rebuild) {
    /* The shape has changed, so need to reconstruct */
    build_links_colloid_fluid(info, map, model, pc);
    if (wall->param->iswall) {
      build_links_colloid_wall(info, map, wall, model, pc);
    }
  }
  else {
    /* Shape unchanged, so just reset existing links */
    build_links_reset_colloid(pc, model, map);
  }

  /* When all links are known, compute ... */

  build_links_evaluate_mean(pc, model);
  build_links_evaluate_area(pc, model);

  pc->s.rebuild = 0;

  return 0;
}

/*****************************************************************************
 *
 *  build_links_update_array_copy
 *
 *****************************************************************************/

int build_links_update_array_copy(colloid_t * pc) {

  assert(pc);
  assert(pc->links);

  int index = 0;

  /* Scrub the existing array (all entries) */

  for (int n = 0; n < pc->links->max_links; n++) {
    pc->links->status[n] = LINK_UNUSED;
  }

  /* Copy over the linked list */

  for (colloid_link_t * lnk = pc->lnk; lnk; lnk = lnk->next, index += 1) {
    colloid_link_to_array(lnk, pc->links, index);
  }

  /* FIXME This should be a run-time failure at some point. */
  pc->links->active_links = index;
  assert(index < pc->links->max_links);

  return 0;
}

/******************************************************************************
 *
 *  build_links_array_colloid_fluid
 *
 *  Device version to wrok with link array. Single thread expected to
 *  run over all links.
 *
 *****************************************************************************/

__host__ __device__ void build_links_array_colloid_fluid(colloids_info_t * info, map_t * map,
				    const lb_model_t * model, colloid_t * pc) {

  const double lambda = 0.5;

  int nlink = 0;

  int i_min, i_max;
  int j_min, j_max;
  int k_min, k_max;
  int nlocal[3] = {};

  double amax = 0.0;

  assert(model);
  assert(pc);

  /* Unset all links */

  for (int n = 0; n < pc->links->max_links; n++) {
    pc->links->status[n] = LINK_UNUSED;
  }

  /* Limits of the search region around the particle. This has to be large
   * enough to capture any local links. Links are "outside to inside" and
   * the outside is not in the halo region. (Such links are captured by the
   * image particle in a neighbouring process.) */

  cs_nlocal(info->cs, nlocal);

  amax = colloid_principal_radius(&pc->s);

  {
    /* Local limits require colloid position minus offset */
    double rc[3] = {};

    rc[X] = pc->s.r[X] - 1.0 * map->cs->param->noffset[X];
    rc[Y] = pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y];
    rc[Z] = pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z];

    i_min = util_imax(1, (int) floor(rc[X] - amax));
    j_min = util_imax(1, (int) floor(rc[Y] - amax));
    k_min = util_imax(1, (int) floor(rc[Z] - amax));
    i_max = util_imin(nlocal[X], (int) ceil(rc[X] + amax));
    j_max = util_imin(nlocal[Y], (int) ceil(rc[Y] + amax));
    k_max = util_imin(nlocal[Z], (int) ceil(rc[Z] + amax));
  }

  /* Begin search ... */

  for (int ic = i_min; ic <= i_max; ic++) {
    for (int jc = j_min; jc <= j_max; jc++) {
      for (int kc = k_min; kc <= k_max; kc++) {

        /* We are looking for links i -> j where i is outside and j is
         * inside; i is a local site (not halo) */

        int    indexi = cs_index(info->cs, ic, jc, kc);
        int    status = MAP_FLUID;
        double r0[3]  = {1.0 * ic, 1.0 * jc, 1.0 * kc};

        colloid_t * pchere = NULL;

        colloids_info_map(info, indexi, &pchere);
        if (pchere == pc) {
          continue;
        }

        map_status(map, indexi, &status);

        /* Index i is outside, so cycle through the lattice vectors
         * to determine if the end is inside, and so requires a link */

        for (int p = 1; p < model->nvel; p++) {

          /* Find the index of the potential inside site */

          int ii = ic + model->cv[p][X];
          int jj = jc + model->cv[p][Y];
          int kk = kc + model->cv[p][Z];

          int    indexj = cs_index(info->cs, ii, jj, kk);
          double rb[3]  = {}; /* centre -> local site (i, j, k) */

          colloids_info_map(info, indexj, &pchere);
          if (pchere != pc) {
            continue;
          }

          /* Index j is inside, so initialise the link */

          rb[X] = r0[X] - (pc->s.r[X] - map->cs->param->noffset[X]);
          rb[Y] = r0[Y] - (pc->s.r[Y] - map->cs->param->noffset[Y]);
          rb[Z] = r0[Z] - (pc->s.r[Z] - map->cs->param->noffset[Z]);

          /* Add the array entry ... */
	  assert(nlink < pc->links->max_links);

	  pc->links->i[nlink]      = indexi;
	  pc->links->j[nlink]      = indexj;
	  pc->links->p[nlink]      = p;
	  pc->links->status[nlink] = LINK_COLLOID;

          if (status == MAP_FLUID) {
	    pc->links->status[nlink] = LINK_FLUID;
          }

	  pc->links->rb[X][nlink] = rb[X] + lambda * model->cv[p][X];
	  pc->links->rb[Y][nlink] = rb[Y] + lambda * model->cv[p][Y];
	  pc->links->rb[Z][nlink] = rb[Z] + lambda * model->cv[p][Z];

	  nlink += 1;
        }

        /* Next site in the search */
      }
    }
  }

  pc->links->active_links = nlink;

  return;
}

/*****************************************************************************
 *
 *  build_links_array_colloid_wall
 *
 *****************************************************************************/

__host__ __device__ void build_links_array_colloid_wall(colloids_info_t * info, map_t * map,
                             wall_t * wall, const lb_model_t * model,
                             colloid_t * pc) {

  const double lambda = 0.5;

  int nlink = 0;

  int i_min, i_max;
  int j_min, j_max;
  int k_min, k_max;
  int nlocal[3] = {};

  double amax = 0.0;

  assert(model);
  assert(pc);

  /* Limits of the search region around the particle. This has to be large
   * enough to capture any local links. Links are "outside to inside" and
   * the outside is not in the halo region. (Such links are captured by the
   * image particle in a neighbouring process.) */

  cs_nlocal(info->cs, nlocal);

  amax = colloid_principal_radius(&pc->s);

  {
    /* Local limits require colloid position minus offset */
    double rc[3] = {};

    rc[X] = pc->s.r[X] - 1.0 * map->cs->param->noffset[X];
    rc[Y] = pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y];
    rc[Z] = pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z];

    i_min = util_imax(1, (int) floor(rc[X] - amax));
    j_min = util_imax(1, (int) floor(rc[Y] - amax));
    k_min = util_imax(1, (int) floor(rc[Z] - amax));
    i_max = util_imin(nlocal[X], (int) ceil(rc[X] + amax));
    j_max = util_imin(nlocal[Y], (int) ceil(rc[Y] + amax));
    k_max = util_imin(nlocal[Z], (int) ceil(rc[Z] + amax));
  }

  /* Begin search ... */

  nlink = pc->links->active_links;

  for (int ic = i_min; ic <= i_max; ic++) {
    int inear = (ic == 1 || ic == nlocal[X]) * wall->param->isboundary[X];
    for (int jc = j_min; jc <= j_max; jc++) {
      int jnear = (jc == 1 || jc == nlocal[Y]) * wall->param->isboundary[Y];
      for (int kc = k_min; kc <= k_max; kc++) {
        int knear = (kc == 1 || kc == nlocal[Z]) * wall->param->isboundary[Z];

        /* We are looking for links i -> j where i is outside and j is
         * inside; i (outside) must be a BOUNDARY */

        int    indexj = cs_index(info->cs, ic, jc, kc);
        double r0[3]  = {1.0 * ic, 1.0 * jc, 1.0 * kc};

        colloid_t * pchere = NULL;

        /* We need to be near the perimeter */
        /* We need indexj inside colloid */
        if (0 == (inear || jnear || knear)) {
          continue;
        }

        colloids_info_map(info, indexj, &pchere);
        if (pchere != pc) {
          continue;
        }

        /* The search is now "inside out" ... */

        for (int p = 1; p < model->nvel; p++) {

          /* Find the index of the potential outside site */

          int ii = ic + model->cv[p][X];
          int jj = jc + model->cv[p][Y];
          int kk = kc + model->cv[p][Z];

          int    indexi = cs_index(info->cs, ii, jj, kk);
          int    status = MAP_FLUID;
          double rb[3]  = {}; /* centre -> local site (i, j, k) */

          map_status(map, indexi, &status);
          if (status != MAP_BOUNDARY) {
            continue;
          }

          /* Index i is boundary, so initialise the link */

          pc->links->i[nlink] = indexi;
          pc->links->j[nlink] = indexj;
          pc->links->p[nlink] = model->nvel - p; /* Opposite direction */
          pc->links->status[nlink] = LINK_BOUNDARY;

          rb[X] = r0[X] - (pc->s.r[X] - map->cs->param->noffset[X]);
          rb[Y] = r0[Y] - (pc->s.r[Y] - map->cs->param->noffset[Y]);
          rb[Z] = r0[Z] - (pc->s.r[Z] - map->cs->param->noffset[Z]);

          pc->links->rb[X][nlink] = rb[X] + lambda * model->cv[p][X];
          pc->links->rb[Y][nlink] = rb[Y] + lambda * model->cv[p][Y];
          pc->links->rb[Z][nlink] = rb[Z] + lambda * model->cv[p][Z];

	  nlink += 1;
        }

        /* Next site in the search */
      }
    }
  }

  pc->links->active_links = nlink;

  return;
}

/*****************************************************************************
 *
 *  build_links_array_reset_colloid
 *
 *  Recompute the boundary link vectors and solid/fluid status
 *  of links for an existing particle.
 *
 ****************************************************************************/

__host__ __device__ void build_links_array_reset_colloid(colloid_t *        pc,
                                                   const lb_model_t * model,
                                                   map_t *            map) {

  const double lambda = 0.5;

  assert(pc);
  assert(map);
  assert(model);

  for (int n = 0; n < pc->links->active_links; n++) {

    if (pc->links->status[n] == LINK_UNUSED) {
      continue;
    }

    /* Compute the separation between the centre of the colloid
     * and the fluid site involved with this link. The position
     * of the outside site is rsite in local coordinates. */

    int i  = pc->links->i[n];
    int p  = pc->links->p[n];

    int ic = cs_index_to_ic(map->cs, i);
    int jc = cs_index_to_jc(map->cs, i);
    int kc = cs_index_to_kc(map->cs, i);

    int status = MAP_COLLOID;

    double r0[3] = {1.0 * ic, 1.0 * jc, 1.0 * kc}; /* this site (local) */
    double rb[3] = {};                             /* colloid centre -> site */

    rb[X] = r0[X] - (pc->s.r[X] - 1.0 * map->cs->param->noffset[X]);
    rb[Y] = r0[Y] - (pc->s.r[Y] - 1.0 * map->cs->param->noffset[Y]);
    rb[Z] = r0[Z] - (pc->s.r[Z] - 1.0 * map->cs->param->noffset[Z]);

    pc->links->rb[X][n] = rb[X] + lambda * model->cv[p][X];
    pc->links->rb[Y][n] = rb[Y] + lambda * model->cv[p][Y];
    pc->links->rb[Z][n] = rb[Z] + lambda * model->cv[p][Z];

    map_status(map, i, &status);

    if (status == MAP_FLUID) {
      pc->links->status[n] = LINK_FLUID;
    }
    else {
      /* Could try to avoid this translation (e.g., make LINK_X = MAP_X) ... */
      if (status == MAP_COLLOID) {
        pc->links->status[n] = LINK_COLLOID;
      }
      if (status == MAP_BOUNDARY) {
        pc->links->status[n] = LINK_BOUNDARY;
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  build_links_array_evaluate_mean
 *
 *  Evaluate sum of weights, and sum of two vector quantities.
 *
 *  Links are a serial loop in the device implementation.
 *
 *****************************************************************************/

__host__ __device__ int build_links_array_evaluate_mean(colloid_t *        pc,
                                                  const lb_model_t * model) {

  /* Evaluate sum of link weights */
  /* Evaluate cbar[] and rxcbar[] */

  pc->sumw = 0.0;

  for (int ia = 0; ia < 3; ia++) {
    pc->cbar[ia]   = 0.0;
    pc->rxcbar[ia] = 0.0;
  }

  /* Only fluid links count ... */

  for (int n = 0; n < pc->links->active_links; n++) {

    if (pc->links->status[n] == LINK_FLUID) {

      int p          = pc->links->p[n];
      double wv      = model->wv[p];
      double wvc[3]  = {};
      double rb[3]   = {};
      double rbxc[3] = {};

      pc->sumw += wv;

      wvc[X] = wv * model->cv[p][X];
      wvc[Y] = wv * model->cv[p][Y];
      wvc[Z] = wv * model->cv[p][Z];

      colloid_links_array_rb(pc->links, n, rb);
      util_vector_cross_product(rbxc, rb, wvc);

      for (int ia = 0; ia < 3; ia++) {
        pc->cbar[ia] += wvc[ia];
        pc->rxcbar[ia] += rbxc[ia];
      }
    }

    /* Next link */
  }

  return 0;
}

/*****************************************************************************
 *
 *  build_links_array_evaluate_area
 *
 *  Count number of faces (local) for this colloid. This is the 'surface
 *  area' on the finite difference grid.
 *
 *  Count both total, and those faces which have fluid neighbours.
 *
 *  Must be serialised on device at the moment.
 *
 *****************************************************************************/

__host__ __device__ int build_links_array_evaluate_area(colloid_t *        pc,
                                                  const lb_model_t * model) {

  assert(pc);
  assert(model);

  pc->s.sa  = 0.0;
  pc->s.saf = 0.0;

  for (int n = 0; n < pc->links->active_links; n++) {
    int p  = pc->links->p[n];
    int pp = util_square_modulus_int8(model->cv[p]);
    if (pp == 1) {
      pc->s.sa += 1.0;
      if (pc->links->status[n] == LINK_FLUID) {
        pc->s.saf += 1.0;
      }
    }
  }

  return 0;
}

/******************************************************************************
 *
 *  build_links_arrray_update_links_colloid
 *
 *  Kernel entry executes one colloid per block (single thread).
 *
 *****************************************************************************/

__host__ __device__ void build_links_array_update_links_colloid(colloids_info_t *  info,
                                     const lb_model_t * model, map_t * map,
						       wall_t * wall, colloid_t * pc) {
  assert(info);
  assert(wall);

  if (pc->s.bc == COLLOID_BC_BBL) {
    printf("KERNAL has %d\n", pc->s.index);
    printf("KERNAL has %f\n", pc->s.r[X]);
    printf("KERNAL has %f\n", pc->s.r[Y]);
    printf("KERNAL has %f\n", pc->s.r[Z]);
    printf("KERNAL has %d\n", pc->s.rebuild);
    if (pc->s.rebuild) {
      /* The shape has changed, so need to reconstruct */
      build_links_array_colloid_fluid(info, map, model, pc);
      if (wall->param->iswall) {
	build_links_array_colloid_wall(info, map, wall, model, pc);
      }
    }
    else {
      /* Shape unchanged, so just reset existing links */
      build_links_array_reset_colloid(pc, model, map);
    }

    /* When all links are known, compute ... */

    build_links_array_evaluate_mean(pc, model);
    build_links_array_evaluate_area(pc, model);

    pc->s.rebuild = 0;
  }

  return;
}

/*****************************************************************************
 *
 *  build_links_array_kernel
 *
 *****************************************************************************/

__global__ void build_links_array_kernel(colloids_info_t * info,
					 const lb_model_t * model,
					 map_t * map,
					 wall_t * wall) {
  assert(info);

  colloid_t * pc = info->pointers->colloid[blockIdx.x];

  build_links_array_update_links_colloid(info, model, map, wall, pc);

  return;
}
  

/*****************************************************************************
 *
 *  build_links_update_driver
 *
 *  Need to look at ther model device version for call from colloids_rt().
 *
 *****************************************************************************/

int build_links_update_driver(colloids_info_t * info, wall_t * wall,
                              map_t * map, const lb_t * lb) {
  assert(info);
  assert(map);
  assert(lb);

  /* Linked-list version */

  for (int n = 0; n < info->npall; n++) {
    colloid_t * pc = info->pointers->colloid[n];
      if (pc->s.bc == COLLOID_BC_BBL) {
      build_links_update_links_colloid(info, &lb->model, map, wall, pc);
    }
  }

  /* Array version */

  int ndevice = 0;

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    /* OpenMP as required */
  }
  else {
    /* One colloid per block, single thread */

    dim3 blocks  = {1, 1, 1};
    dim3 threads = {1, 1, 1};

    wall_t * target = NULL;
    if (wall) target = wall->target;

    blocks.x = info->npall;

    tdpLaunchKernel(build_links_array_kernel, blocks, threads, 0, 0,
		    info->target, &lb->target->model, map->target, target);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpStreamSynchronize(0));
  }

  return 0;
}
