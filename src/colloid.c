/*****************************************************************************
 *
 *  colloid.c
 *
 *  State type for particles including bounce-back on links, wetting,
 *  magnetic dipoles, and squirmers.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "util_ellipsoid.h"
#include "colloid.h"

/* Old type definitions for backwards compatibility */
enum colloid_type_enum {COLLOID_TYPE_DEFAULT = 0,
                        COLLOID_TYPE_ACTIVE,
                        COLLOID_TYPE_SUBGRID,
                        COLLOID_TYPE_JANUS};

/*****************************************************************************
 *
 *  colloid_state_read_ascii
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int colloid_state_read_ascii(colloid_state_t * ps, FILE * fp) {

  int n;
  int nread = 0;
  int ifail = 0;

  const char * isformat = "%24d\n";
  const char * sformat  = "%24le\n";
  const char * vformat  = "%24le %24le %24le\n";

  assert(ps);
  assert(fp);

  nread += fscanf(fp, isformat, &ps->index);
  nread += fscanf(fp, isformat, &ps->rebuild);
  nread += fscanf(fp, isformat, &ps->nbonds);
  nread += fscanf(fp, isformat, &ps->nangles);
  nread += fscanf(fp, isformat, &ps->isfixedr);
  nread += fscanf(fp, isformat, &ps->isfixedv);
  nread += fscanf(fp, isformat, &ps->isfixedw);
  nread += fscanf(fp, isformat, &ps->isfixeds);
  nread += fscanf(fp, isformat, &ps->type);

  for (n = 0; n < NBOND_MAX; n++) {
    nread += fscanf(fp, isformat, &ps->bond[n]);
  }

  nread += fscanf(fp, isformat, &ps->rng);

  for (n = 0; n < 3; n++) {
    nread += fscanf(fp, isformat, ps->isfixedrxyz + n);
  }
  for (n = 0; n < 3; n++) {
    nread += fscanf(fp, isformat, ps->isfixedvxyz + n);
  }

  nread += fscanf(fp, isformat, &ps->inter_type);
  nread += fscanf(fp, isformat, &ps->ioversion);
  nread += fscanf(fp, isformat, &ps->bc);
  nread += fscanf(fp, isformat, &ps->shape);
  nread += fscanf(fp, isformat, &ps->active);
  nread += fscanf(fp, isformat, &ps->magnetic);
  nread += fscanf(fp, isformat, &ps->attr);

  for (n = 0; n < NPAD_INT; n++) {
    nread += fscanf(fp, isformat, &ps->intpad[n]);
  }

  nread += fscanf(fp, sformat, &ps->a0);
  nread += fscanf(fp, sformat, &ps->ah);
  nread += fscanf(fp, vformat, &ps->r[0], &ps->r[1], &ps->r[2]);
  nread += fscanf(fp, vformat, &ps->v[0], &ps->v[1], &ps->v[2]);
  nread += fscanf(fp, vformat, &ps->w[0], &ps->w[1], &ps->w[2]);
  nread += fscanf(fp, vformat, &ps->s[0], &ps->s[1], &ps->s[2]);
  nread += fscanf(fp, vformat, &ps->m[0], &ps->m[1], &ps->m[2]);
  nread += fscanf(fp, sformat, &ps->b1);
  nread += fscanf(fp, sformat, &ps->b2);
  nread += fscanf(fp, sformat, &ps->c);
  nread += fscanf(fp, sformat, &ps->h);
  nread += fscanf(fp, vformat, &ps->dr[0], &ps->dr[1], &ps->dr[2]);
  nread += fscanf(fp, sformat, &ps->deltaphi);

  nread += fscanf(fp, sformat, &ps->q0);
  nread += fscanf(fp, sformat, &ps->q1);
  nread += fscanf(fp, sformat, &ps->epsilon);

  nread += fscanf(fp, sformat, &ps->deltaq0);
  nread += fscanf(fp, sformat, &ps->deltaq1);
  nread += fscanf(fp, sformat, &ps->sa);
  nread += fscanf(fp, sformat, &ps->saf);
  nread += fscanf(fp, sformat, &ps->al);

  /* For backwards compatibility, these are read one line at a time */
  nread += fscanf(fp, sformat, &ps->elabc[0]);
  nread += fscanf(fp, sformat, &ps->elabc[1]);
  nread += fscanf(fp, sformat, &ps->elabc[2]);

  nread += fscanf(fp, sformat, &ps->quat[0]);
  nread += fscanf(fp, sformat, &ps->quat[1]);
  nread += fscanf(fp, sformat, &ps->quat[2]);
  nread += fscanf(fp, sformat, &ps->quat[3]);

  nread += fscanf(fp, sformat, &ps->quatold[0]);
  nread += fscanf(fp, sformat, &ps->quatold[1]);
  nread += fscanf(fp, sformat, &ps->quatold[2]);
  nread += fscanf(fp, sformat, &ps->quatold[3]);

  for (n = 0; n < NPAD_DBL; n++) {
    nread += fscanf(fp, sformat, &ps->dpad[n]);
  }

  if (nread != NTOT_VAR) ifail = 1;

  /* If assertions are off, we may want to catch this failure elsewhere */
  assert(ifail == 0);

  /* Always set the rebuild flag (even if file has zero) */

  ps->rebuild = 1;

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_state_read_binary
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int colloid_state_read_binary(colloid_state_t * ps, FILE * fp) {

  int nread;

  assert(ps);
  assert(fp);

  nread = fread(ps, sizeof(colloid_state_t), 1, fp);

  /* Always set the rebuild flag (even if file has zero) */

  ps->rebuild = 1;

  return (1 - nread);
}

/*****************************************************************************
 *
 *  colloid_state_write_ascii
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int colloid_state_write_ascii(const colloid_state_t * s, FILE * fp) {

  int n;
  int nwrite = 0;
  int ifail = 0;

  const char * isformat = "%24d\n";
  const char * sformat  = "%24.15e\n";
  const char * vformat  = "%24.15e %24.15e %24.15e\n";

  assert(s);
  assert(fp);

  nwrite += fprintf(fp, isformat, s->index);
  nwrite += fprintf(fp, isformat, s->rebuild);
  nwrite += fprintf(fp, isformat, s->nbonds);
  nwrite += fprintf(fp, isformat, s->nangles);
  nwrite += fprintf(fp, isformat, s->isfixedr);
  nwrite += fprintf(fp, isformat, s->isfixedv);
  nwrite += fprintf(fp, isformat, s->isfixedw);
  nwrite += fprintf(fp, isformat, s->isfixeds);
  nwrite += fprintf(fp, isformat, s->type);

  for (n = 0; n < NBOND_MAX; n++) {
    nwrite += fprintf(fp, isformat, s->bond[n]);
  }

  nwrite += fprintf(fp, isformat, s->rng);

  /* isfixedrxyz and isfixedvxyz are written as 3 x scalars as they
   * have replaced padding */

  for (n = 0; n < 3; n++) {
    nwrite += fprintf(fp, isformat, s->isfixedrxyz[n]);
  }
  for (n = 0; n < 3; n++) {
    nwrite += fprintf(fp, isformat, s->isfixedvxyz[n]);
  }

  nwrite += fprintf(fp, isformat, s->inter_type);
  nwrite += fprintf(fp, isformat, s->ioversion);
  nwrite += fprintf(fp, isformat, s->bc);
  nwrite += fprintf(fp, isformat, s->shape);
  nwrite += fprintf(fp, isformat, s->active);
  nwrite += fprintf(fp, isformat, s->magnetic);
  nwrite += fprintf(fp, isformat, s->attr);

  for (n = 0; n < NPAD_INT; n++) {
    nwrite += fprintf(fp, isformat, s->intpad[n]);
  }

  nwrite += fprintf(fp, sformat, s->a0);
  nwrite += fprintf(fp, sformat, s->ah);
  nwrite += fprintf(fp, vformat, s->r[0], s->r[1], s->r[2]);
  nwrite += fprintf(fp, vformat, s->v[0], s->v[1], s->v[2]);
  nwrite += fprintf(fp, vformat, s->w[0], s->w[1], s->w[2]);
  nwrite += fprintf(fp, vformat, s->s[0], s->s[1], s->s[2]);
  nwrite += fprintf(fp, vformat, s->m[0], s->m[1], s->m[2]);
  nwrite += fprintf(fp, sformat, s->b1);
  nwrite += fprintf(fp, sformat, s->b2);
  nwrite += fprintf(fp, sformat, s->c);
  nwrite += fprintf(fp, sformat, s->h);
  nwrite += fprintf(fp, vformat, s->dr[0], s->dr[1], s->dr[2]);
  nwrite += fprintf(fp, sformat, s->deltaphi);

  nwrite += fprintf(fp, sformat, s->q0);
  nwrite += fprintf(fp, sformat, s->q1);
  nwrite += fprintf(fp, sformat, s->epsilon);

  nwrite += fprintf(fp, sformat, s->deltaq0);
  nwrite += fprintf(fp, sformat, s->deltaq1);
  nwrite += fprintf(fp, sformat, s->sa);
  nwrite += fprintf(fp, sformat, s->saf);
  nwrite += fprintf(fp, sformat, s->al);

  /* Additional entries should be one data item per line at a time */

  nwrite += fprintf(fp, sformat, s->elabc[0]);
  nwrite += fprintf(fp, sformat, s->elabc[1]);
  nwrite += fprintf(fp, sformat, s->elabc[2]);

  nwrite += fprintf(fp, sformat, s->quat[0]);
  nwrite += fprintf(fp, sformat, s->quat[1]);
  nwrite += fprintf(fp, sformat, s->quat[2]);
  nwrite += fprintf(fp, sformat, s->quat[3]);

  nwrite += fprintf(fp, sformat, s->quatold[0]);
  nwrite += fprintf(fp, sformat, s->quatold[1]);
  nwrite += fprintf(fp, sformat, s->quatold[2]);
  nwrite += fprintf(fp, sformat, s->quatold[3]);

  /* Padding */

  for (n = 0; n < NPAD_DBL; n++) {
    nwrite += fprintf(fp, sformat, s->dpad[n]);
  }

  /* ... should be NTOT_VAR items of format + 1 characters */
  if (nwrite != NTOT_VAR*25) ifail = 1;

  /* If assertions are off, responsibility passes to caller */
  assert(ifail == 0);

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_state_write_binary
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int colloid_state_write_binary(const colloid_state_t * s, FILE * fp) {

  int nwrite;

  assert(s);
  assert(fp);

  nwrite = fwrite(s, sizeof(colloid_state_t), 1, fp);

  return (1 - nwrite);
}

/*****************************************************************************
 *
 *  colloid_state_mass
 *
 *  Depends on shape and density.
 *
 *****************************************************************************/

int colloid_state_mass(const colloid_state_t * s, double rho0, double * mass) {

  int ifail = 0;
  const double pi = 4.0*atan(1.0);

  assert(s);
  assert(mass);

  if (s->shape == COLLOID_SHAPE_SPHERE) {
    *mass = 4.0*pi*pow(s->a0, 3)*rho0/3.0;
  }
  else if (s->shape == COLLOID_SHAPE_ELLIPSOID) {
    *mass = (4.0/3.0)*pi*rho0*s->elabc[0]*s->elabc[1]*s->elabc[2];
  }
  else if (s->shape == COLLOID_SHAPE_DISK) {
    *mass = pi*rho0*pow(s->a0, 2);
  }
  else {
    ifail = -1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_type_check
 *
 *  At v0.21.0 colloid.type was effectively split into a number of separate
 *  components "shape" "bc" "active" etc.
 *  If the latest version reads an old format file with shape defaulting
 *  to COLLOID_SHAPE_INVALID, we must recover reasonable behaviour.
 *
 *  So assume we should have the old default situation with a sphere
 *  using bbl.
 *
 *****************************************************************************/

int colloid_type_check(colloid_state_t * s) {

  int is_updated = 0;

  if (s->shape == COLLOID_SHAPE_INVALID) {
    s->shape   = COLLOID_SHAPE_SPHERE;
    s->bc      = COLLOID_BC_BBL;
    s->active  = 0;
    if (s->type == COLLOID_TYPE_ACTIVE) s->active = 1;
    if (s->type == COLLOID_TYPE_SUBGRID) s->bc = COLLOID_BC_SUBGRID;
    is_updated = 1;
  }

  return is_updated;
}

/*****************************************************************************
 *
 *  colloid_principal_radius
 *
 *  The radius; in the case of an ellipsoid, the principal a (a >= b >= c).
 *
 *****************************************************************************/

double colloid_principal_radius(const colloid_state_t * s) {

  double amax = -1.0;

  assert(s);

  amax = s->a0;
  if (s->shape == COLLOID_SHAPE_ELLIPSOID) amax = s->elabc[0];

  return amax;
}

/*****************************************************************************
 *
 *  colloid_r_inside
 *
 *  Is r inside the colloid? The vector r is a displacement from the centre.
 *  For details of the ellipsoid case, see util_4_is_inside_ellipsoid().
 *
 *  Return value of -1 is an error.
 *
 *****************************************************************************/

int colloid_r_inside(const colloid_state_t * s, const double r[3]) {

  int inside = 0;

  if (s->shape == COLLOID_SHAPE_SPHERE) {
    double rdot = r[X]*r[X] + r[Y]*r[Y] + r[Z]*r[Z];
    if (rdot < s->a0*s->a0) inside = 1;
  }
  else if (s->shape == COLLOID_SHAPE_ELLIPSOID) {
    inside = util_q4_is_inside_ellipsoid(s->quat, s->elabc, r);
  }
  else if (s->shape == COLLOID_SHAPE_DISK) {
    double rdot = r[X]*r[X] + r[Y]*r[Y];
    if (rdot < s->a0*s->a0) inside = 1;
  }
  else {
    /* This should have been trapped at input. */
    inside = -1;
  }

  return inside;
}
