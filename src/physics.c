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
 *  Note for Developers: the singleton object here is not doing any
 *  favours (particularly for device code). I would like to move to
 *  a position where the object is passed explicitly. This reduces
 *  scope for surprise, particluarly when writing unit tests.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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

  double fpulse_frequency; /* Frequency of external body forcing */
  double fpulse[3];        /* Amplitude of external electric field */

  double grad_mu[3];       /* External chemical potential gradient */
};

/* At the moment we have static instances */

static physics_t * stat_phys = NULL;
static __constant__ physics_t const_phys;

/*****************************************************************************
 *
 *  physics_ref
 *
 *  Return an appropriate reference.
 *
 *****************************************************************************/

__host__ __device__ int physics_ref(physics_t ** ref) {

  assert(ref);

#if defined( __CUDA_ARCH__ ) || defined (__HIP_DEVICE_COMPILE__)
  *ref = &const_phys;
#else
  assert(stat_phys);
  *ref = stat_phys;
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

__host__ int physics_create(pe_t * pe, physics_t ** phys) {

  physics_t * obj = NULL;

  assert(pe);
  assert(phys);

  obj = (physics_t *) calloc(1, sizeof(physics_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(physics_t) failed\n");

  obj->eta_shear = ETA_DEFAULT;
  obj->eta_bulk  = ETA_DEFAULT;
  obj->rho0      = RHO_DEFAULT;

  /* Everything else defaults to zero. */

  /* Time control */

  obj->nsteps = 0;
  obj->t_start = 0;
  obj->t_current = 0;

  if (stat_phys == NULL) stat_phys = obj;
  *phys = obj;

  return 0;
}

/*****************************************************************************
 *
 *  physics_free
 *
 *****************************************************************************/

__host__ int physics_free(physics_t * phys) {

  assert(phys);

  free(phys);
  stat_phys = NULL;

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

  tdpMemcpyToSymbol(tdpSymbol(const_phys), phys, sizeof(physics_t), 0,
		    tdpMemcpyHostToDevice);

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

__host__ __device__ int physics_fbody(const physics_t * phys, double f[3]) {

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
 *  physics_fpulse
 *
 *****************************************************************************/

__host__ __device__ int physics_fpulse(physics_t * phys, double fpulse[3]) {

  assert(phys);

  fpulse[0] = phys->fpulse[0];
  fpulse[1] = phys->fpulse[1];
  fpulse[2] = phys->fpulse[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_fpulse_set
 *
 *****************************************************************************/

__host__ int physics_fpulse_set(physics_t * phys, double fpulse[3]) {

  assert(phys);

  phys->fpulse[0] = fpulse[0];
  phys->fpulse[1] = fpulse[1];
  phys->fpulse[2] = fpulse[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_fpulse_frequency
 *
 *****************************************************************************/

__host__ __device__ int physics_fpulse_frequency(physics_t * phys,
					     double * fpulse_frequency) {

  assert(phys);

  *fpulse_frequency = phys->fpulse_frequency;

  return 0;
}

/*****************************************************************************
 *
 *  physics_fpulse_frequency_set
 *
 *****************************************************************************/

__host__ int physics_fpulse_frequency_set(physics_t * phys, double fpulse_frequency) {

  assert(phys);

  phys->fpulse_frequency = fpulse_frequency;

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

__host__ __device__ int physics_fgrav(const physics_t * phys, double g[3]) {

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
 *  physics_grad_mu - added for externally imposed chemical potential gradient
 *
 *****************************************************************************/

__host__ __device__ int physics_grad_mu(physics_t * phys, double gm[3]) {

  assert(phys);

  gm[0] = phys->grad_mu[0];
  gm[1] = phys->grad_mu[1];
  gm[2] = phys->grad_mu[2];

  return 0;
}

/*****************************************************************************
 *
 *  physics_grad_mu_set - added for externally imposed chemical potential gradient
 *
 *****************************************************************************/

__host__ int physics_grad_mu_set(physics_t * phys, double gm[3]) {

  assert(phys);

  phys->grad_mu[0] = gm[0];
  phys->grad_mu[1] = gm[1];
  phys->grad_mu[2] = gm[2];

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
