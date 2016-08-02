/*****************************************************************************
 *
 *  physics.c
 *
 *  These data are broadly physical properties of the fluid, constant
 *  external fields, and so on.
 *
 *  The host carries the true and correct current values of these
 *  constants, including those related to time-stepping.
 *
 *  Device copy must be updated at least once per time step to
 *  ensure a coherent view.
 *
 *
 *  $Id: physics.c,v 1.4 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
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

  int t_start;         /* Start time step requested */
  int nsteps;          /* Number of time steps requested by user */
  int t_current;       /* The current time step */
  int e0_flag;         /* = 1 if electric field is non-zero */
};

/* At the moment we have static instances */

static physics_t * phys = NULL;
static __constant__ physics_t const_phys;

__host__ int physics_create(void);

/*****************************************************************************
 *
 *  physics_ref
 *
 *  Return an appropriate reference.
 *
 *****************************************************************************/

__host__ __device__ int physics_ref(physics_t ** ref) {

  assert(ref);

#ifdef __CUDA_ARCH__
  *ref = &const_phys;
#else
  if (phys == NULL) physics_create();

  *ref = phys;
#endif

  return 0;
}

/*****************************************************************************
 *
 *  physics_create
 *
 *  A single static object.
 *
 *****************************************************************************/

__host__ int physics_create(void) {

  assert(phys == NULL);

  phys = (physics_t *) calloc(1, sizeof(physics_t));
  if (phys == NULL) fatal("calloc(physics_t) failed\n");

  phys->eta_shear = ETA_DEFAULT;
  phys->eta_bulk  = ETA_DEFAULT;
  phys->rho0      = RHO_DEFAULT;

  /* Everything else defaults to zero. */

  /* Time control */

  phys->nsteps = 0;
  phys->t_start = 0;
  phys->t_current = 0;

  return 0;
}

/*****************************************************************************
 *
 *  physics_free
 *
 *****************************************************************************/

__host__ int physics_free(void) {

  assert(phys);

  free(phys);
  phys = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  physics_commit
 *
 *  To be called at least once per timestep.
 *
 *****************************************************************************/

__host__ int physics_commit(physics_t * phys) {

  assert(phys);

  copyConstToTarget(&const_phys, phys, sizeof(physics_t));

  return 0;
}

/*****************************************************************************
 *
 *  physics_eta_shear
 *
 *****************************************************************************/

__host__ __device__ int physics_eta_shear(physics_t * phys, double * eta) {

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

__host__ int physics_eta_shear_set(physics_t * phys, double eta) {

  assert(phys);

  phys->eta_shear = eta;

  return 0;
}

/*****************************************************************************
 *
 *  physics_eta_bulk
 *
 *****************************************************************************/

__host__ __device__ int physics_eta_bulk(physics_t * phys, double * eta) {

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

__host__ int physics_eta_bulk_set(physics_t * phys, double eta) {

  assert(phys);

  phys->eta_bulk = eta;

  return 0;
}

/*****************************************************************************
 *
 *  physics_rho0
 *
 *****************************************************************************/

__host__ __device__ int physics_rho0(physics_t * phys, double * rho0) {

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

__host__ int physics_rho0_set(physics_t * phys, double rho0) {

  assert(phys);

  phys->rho0 = rho0;

  return 0;
}

/*****************************************************************************
 *
 *  physics_kt
 *
 *****************************************************************************/

__host__ __device__ int physics_kt(physics_t * phys, double * kt) {

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

__host__ int physics_kt_set(physics_t * phys, double kt) {

  assert(phys);

  phys->kt = kt;

  return 0;
}

/*****************************************************************************
 *
 *  physics_phi0
 *
 *****************************************************************************/

__host__ __device__ int physics_phi0(physics_t * phys, double * phi0) {

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

__host__ int physics_phi0_set(physics_t * phys, double phi0) {

  assert(phys);

  phys->phi0 = phi0;

  return 0;
}

/*****************************************************************************
 *
 *  physics_b0
 *
 *****************************************************************************/

__host__ __device__ int physics_b0(physics_t * phys, double b0[3]) {

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

__host__ int physics_b0_set(physics_t * phys, double b0[3]) {

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

__host__ __device__ int physics_e0(physics_t * phys, double e0[3]) {

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

__host__ int physics_e0_set(physics_t * phys, double e0[3]) {

  assert(phys);

  phys->e0[0] = e0[0];
  phys->e0[1] = e0[1];
  phys->e0[2] = e0[2];

  if (e0[0] != 0.0 || e0[1] != 0.0 || e0[2] != 0.0) phys->e0_flag = 1;

  return 0;
}

/*****************************************************************************
 *
 *  physics_e0_flag
 *
 *  Returns flag if external electric field is set. This is required 
 *  when constraints are applied in the Poisson solve to generate
 *  a potential jump.
 *
 *****************************************************************************/

__host__ __device__ int physics_e0_flag(physics_t * phys) {

  assert(phys);

  return phys->e0_flag;
}

/*****************************************************************************
 *
 *  physics_e0_frequency
 *
 *****************************************************************************/

__host__ __device__ int physics_e0_frequency(physics_t * phys,
					     double * e0_frequency) {

  assert(phys);

  *e0_frequency = phys->e0_frequency;

  return 0;
}

/*****************************************************************************
 *
 *  physics_e0_frequency_set
 *
 *****************************************************************************/

__host__ int physics_e0_frequency_set(physics_t * phys, double e0_frequency) {

  assert(phys);

  phys->e0_frequency = e0_frequency;

  return 0;
}

/*****************************************************************************
 *
 *  physics_fbody
 *
 *****************************************************************************/

__host__ __device__ int physics_fbody(physics_t * phys, double f[3]) {

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

__host__ int physics_fbody_set(physics_t * phys, double f[3]) {

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

__host__ __device__ int physics_mobility(physics_t * phys, double * mobility) {

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

__host__ int physics_mobility_set(physics_t * phys, double mobility) {

  assert(mobility);

  phys->mobility = mobility;

  return 0;
}

/*****************************************************************************
 *
 *  physics_fgrav
 *
 *****************************************************************************/

__host__ __device__ int physics_fgrav(physics_t * phys, double g[3]) {

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

__host__ int physics_fgrav_set(physics_t * phys, double g[3]) {

  assert(phys);

  phys->fgravity[0] = g[0];
  phys->fgravity[1] = g[1];
  phys->fgravity[2] = g[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_control_timestep
 *
 *  Time step counter always counts 1... nsteps
 *
 *****************************************************************************/

__host__ __device__ int physics_control_timestep(physics_t * phys) {

  assert(phys);
  return phys->t_current;
}

/*****************************************************************************
 *
 *  physics_control_time
 *
 *  Return the current "time" in lattice units dt = 1.0.
 *  In the first iteration, the time is 1.0, not 0.0 or 0.5.
 *  Thus the -1.0.
 *
 *****************************************************************************/

__host__ __device__ int physics_control_time(physics_t * phys, double * t) {

  assert(phys);
  assert(t);

  *t = 1.0*(phys->t_start + phys->t_current - 1.0);

  return 0;
}

/*****************************************************************************
 *
 *  physics_control_next_step
 *
 *  This is the only way the time step should be incremented.
 *  Returns 0 if there are no more steps to be taken.
 *
 *****************************************************************************/

__host__ int physics_control_next_step(physics_t * phys) {

  assert(phys);

  phys->t_current += 1;
  return (phys->t_start + phys->nsteps - phys->t_current + 1);
}

/*****************************************************************************
 *
 *  physics_control_timestep_set
 *
 *  At the start of execution, current time should be set to start
 *  time.
 *
 *****************************************************************************/

__host__ int physics_control_init_time(physics_t * phys, int nstart, int nstep) {

  assert(phys);

  phys->t_start = nstart;
  phys->nsteps = nstep;

  phys->t_current = nstart;

  return 0;
}
