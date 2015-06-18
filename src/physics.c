/*****************************************************************************
 *
 *  physics.c
 *
 *  These data are broadly physical properties of the fluid, constant
 *  external fields, and so on.
 *
 *  $Id: physics.c,v 1.4 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "physics.h"

#define ETA_DEFAULT (1.0/6.0)
#define RHO_DEFAULT 1.0

struct physics_s {
  double eta_shear;    /* Shear viscosity */
  double eta_bulk;     /* Bulk viscosity */
  double kt;           /* Isothermal "temperature" */
  double rho0;         /* Mean fluid density */
  double phi0;         /* Mean fluid composition (binary fluid) */
  double phi_noise0;   /* Initial order parameter noise amplitude */
  double fbody[3];     /* External body force on fluid */
  double e0[3];        /* Amplitude of external electric field */
  double e0_frequency; /* Frequency of external electric field */ 
  double b0[3];        /* External magnetic field */
  double fgravity[3];  /* Gravitational force (on objects) */
  double mobility;     /* Order parameter mobility (binary fluid) */
  double lc_gamma_rot; /* Liquid crystal rotational diffusion coefficient */
};

static physics_t * phys = NULL;
static int physics_create(void);

static int e0_flag = 0;

/*****************************************************************************
 *
 *  physics_ref
 *
 *****************************************************************************/

int physics_ref(physics_t ** ref) {

  assert(ref);

  if (phys == NULL) physics_create();

  *ref = phys;

  return 0;
}

/*****************************************************************************
 *
 *  physics_create
 *
 *  A single static object.
 *
 *****************************************************************************/

static int physics_create(void) {

  assert(phys == NULL);

  phys = (physics_t *) calloc(1, sizeof(physics_t));
  if (phys == NULL) fatal("calloc(physics_t) failed\n");

  phys->eta_shear = ETA_DEFAULT;
  phys->eta_bulk  = ETA_DEFAULT;
  phys->rho0      = RHO_DEFAULT;

  /* Everything else defaults to zero. */

  return 0;
}

/*****************************************************************************
 *
 *  physics_free
 *
 *****************************************************************************/

int physics_free(void) {

  assert(phys);

  free(phys);
  phys = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  physics_eta_shear
 *
 *****************************************************************************/

int physics_eta_shear(double * eta) {

  assert(phys);
  assert(eta);

  *eta = phys->eta_shear;

  return 0;
}

/*****************************************************************************
 *
 *  physics_eta_shear_set
 *
 *****************************************************************************/

int physics_eta_shear_set(double eta) {

  assert(phys);

  phys->eta_shear = eta;

  return 0;
}

/*****************************************************************************
 *
 *  physics_eta_bulk
 *
 *****************************************************************************/

int physics_eta_bulk(double * eta) {

  assert(phys);
  assert(eta);

  *eta = phys->eta_bulk;

  return 0;
}

/*****************************************************************************
 *
 *  physics_eta_bulk_set
 *
 *****************************************************************************/

int physics_eta_bulk_set(double eta) {

  assert(phys);

  phys->eta_bulk = eta;

  return 0;
}

/*****************************************************************************
 *
 *  physics_rho0
 *
 *****************************************************************************/

int physics_rho0(double * rho0) {

  assert(phys);
  assert(rho0);

  *rho0 = phys->rho0;

  return 0;
}

/*****************************************************************************
 *
 *  physics_rho0_set
 *
 *****************************************************************************/

int physics_rho0_set(double rho0) {

  assert(phys);

  phys->rho0 = rho0;

  return 0;
}

/*****************************************************************************
 *
 *  physics_kt
 *
 *****************************************************************************/

int physics_kt(double * kt) {

  assert(phys);
  assert(kt);

  *kt = phys->kt;

  return 0;
}

/*****************************************************************************
 *
 *  physics_kt_set
 *
 *****************************************************************************/

int physics_kt_set(double kt) {

  assert(phys);

  phys->kt = kt;

  return 0;
}

/*****************************************************************************
 *
 *  physics_phi0
 *
 *****************************************************************************/

int physics_phi0(double * phi0) {

  assert(phys);
  assert(phi0);

  *phi0 = phys->phi0;

  return 0;
}

/*****************************************************************************
 *
 *  physics_phi0_set
 *
 *****************************************************************************/

int physics_phi0_set(double phi0) {

  assert(phys);

  phys->phi0 = phi0;

  return 0;
}

/*****************************************************************************
 *
 *  physics_b0
 *
 *****************************************************************************/

int physics_b0(double b0[3]) {

  assert(phys);

  b0[0] = phys->b0[0];
  b0[1] = phys->b0[1];
  b0[2] = phys->b0[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_b0_set
 *
 *****************************************************************************/

int physics_b0_set(double b0[3]) {

  assert(phys);

  phys->b0[0] = b0[0];
  phys->b0[1] = b0[1];
  phys->b0[2] = b0[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_e0
 *
 *****************************************************************************/

int physics_e0(double e0[3]) {

  assert(phys);

  e0[0] = phys->e0[0];
  e0[1] = phys->e0[1];
  e0[2] = phys->e0[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_e0_set
 *
 *****************************************************************************/

int physics_e0_set(double e0[3]) {

  assert(phys);

  phys->e0[0] = e0[0];
  phys->e0[1] = e0[1];
  phys->e0[2] = e0[2];

  if (e0[0] != 0.0 || e0[1] != 0.0 || e0[2] != 0.0) e0_flag = 1;

  return 0;
}

/*****************************************************************************
 *
 *  is_physics_e0
 *
 *  Returns flag if external electric field is set. This is required 
 *  when constraints are applied in the Poisson solve to generate
 *  a potential jump.
 *
 *****************************************************************************/

int is_physics_e0() {

  assert(phys);

  return e0_flag;
}

/*****************************************************************************
 *
 *  physics_e0_frequency
 *
 *****************************************************************************/

int physics_e0_frequency(double * e0_frequency) {

  assert(phys);

  *e0_frequency = phys->e0_frequency;

  return 0;
}

/*****************************************************************************
 *
 *  physics_e0_frequency_set
 *
 *****************************************************************************/

int physics_e0_frequency_set(double e0_frequency) {

  assert(phys);

  phys->e0_frequency = e0_frequency;

  return 0;
}

/*****************************************************************************
 *
 *  physics_fbody
 *
 *****************************************************************************/

int physics_fbody(double f[3]) {

  assert(phys);

  f[0] = phys->fbody[0];
  f[1] = phys->fbody[1];
  f[2] = phys->fbody[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_fbody_set
 *
 *****************************************************************************/

int physics_fbody_set(double f[3]) {

  assert(phys);

  phys->fbody[0] = f[0];
  phys->fbody[1] = f[1];
  phys->fbody[2] = f[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_mobility
 *
 *****************************************************************************/

int physics_mobility(double * mobility) {

  assert(phys);
  assert(mobility);

  *mobility = phys->mobility;
 
  return 0;
}

/*****************************************************************************
 *
 *  physics_mobility_set
 *
 *****************************************************************************/

int physics_mobility_set(double mobility) {

  assert(mobility);

  phys->mobility = mobility;

  return 0;
}

/*****************************************************************************
 *
 *  physics_lc_gamma_rot
 *
 *****************************************************************************/

int physics_lc_gamma_rot(double * gamma) {

  assert(phys);
  assert(gamma);

  *gamma = phys->lc_gamma_rot;

  return 0;
}


/*****************************************************************************
 *
 *  physics_lc_gamma_rot_set
 *
 *****************************************************************************/

int physics_lc_gamma_rot_set(double gamma) {

  assert(phys);

  phys->lc_gamma_rot = gamma;

  return 0;
}

/*****************************************************************************
 *
 *  physics_fgrav
 *
 *****************************************************************************/

int physics_fgrav(double g[3]) {

  assert(phys);

  g[0] = phys->fgravity[0];
  g[1] = phys->fgravity[1];
  g[2] = phys->fgravity[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_fgrav_set
 *
 *****************************************************************************/

int physics_fgrav_set(double g[3]) {

  assert(phys);

  phys->fgravity[0] = g[0];
  phys->fgravity[1] = g[1];
  phys->fgravity[2] = g[2];

  return 0;
}
