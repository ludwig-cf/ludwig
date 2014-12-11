/*****************************************************************************
 *
 *  stats_calibration.c
 *
 *  A measurement to calibrate the hydrodynamic radius.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <math.h>

#include "util.h"
#include "coords.h"
#include "physics.h"
#include "stats_calibration.h"

#define TARGET_REYNOLDS_NUMBER 0.05
#define MEASUREMENTS_PER_STOKES_TIME 50

struct stats_calibration_type {
  int nstart;
  int ndata;
  int nstokes;
  int nfreq;
  double a0;
  double utarget;
  double ftarget;
  double ubar[3];
  double fbar[3];
};

typedef struct stats_calibration_type stats_calibration_t;

static stats_calibration_t calib_;
static double stats_calibration_hasimoto(double a, double length);
static int stats_calibration_measure(colloids_info_t * cinfo, hydro_t * hydro,
				     map_t * map);

/*****************************************************************************
 *
 *  stats_calibration_init
 *
 *  1. We control the particle Reynolds number rho a U / eta (= 0.05)
 *     to give a target velocity.
 *  2. This gives us a estimate of the force required to drive
 *     sedimentation at this velocity.
 *  3. This velocity sets the Stokes' time a/U.
 *  4. Allow a spin-up period equal to the momentum diffusion time
 *     for the whole system L^2 / eta. This is nstart.
 *  5. After this, we can take measurements of the mean force on
 *     the particle fbar, and the mean velocity.
 *  6. These are accumulated with count ndata.
 *  7. At the end we estimate the hydrodynamic radius.
 *
 *****************************************************************************/

int stats_calibration_init(colloids_info_t * cinfo, int nswitch) {

  int ia;
  int nc;
  int ntotal[3];
  int state = 13;
  double a;
  double rho;
  double eta;
  double length;
  double fhasimoto;
  double f[3];

  if (nswitch == 0) {
    /* No statistics are required */
    calib_.nstart = INT_MAX;
    calib_.nfreq = INT_MAX;
  }
  else {

    assert(cinfo);

    coords_ntotal(ntotal);

    /* Make sure we have a cubic system with one particle */

    if (ntotal[X] != ntotal[Y]) fatal("Calibration must have cubic system\n");
    if (ntotal[Y] != ntotal[Z]) fatal("Calibration must have cubic system\n");

    colloids_info_ntotal(cinfo, &nc);
    if (nc != 1) fatal("Calibration requires exactly one colloid\n");

    length = 1.0*L(Z);
    physics_rho0(&rho);
    physics_eta_shear(&eta);

    colloids_info_ahmax(cinfo, &a);

    calib_.a0 = a;
    calib_.utarget = eta*TARGET_REYNOLDS_NUMBER/(a*rho);
    fhasimoto = stats_calibration_hasimoto(a, length);
    calib_.ftarget = 6.0*pi_*eta*a*calib_.utarget/fhasimoto;

    calib_.nstokes = a/calib_.utarget;
    calib_.nfreq = calib_.nstokes/MEASUREMENTS_PER_STOKES_TIME;
    if (calib_.nfreq < 1) calib_.nfreq = 1;
    calib_.nstart = length*length/eta;

    /* Set a force of the right size in a random direction, and zero
     * the accumulators. */

    util_ranlcg_reap_unit_vector(&state, f);
    /*ran_serial_unit_vector(f);*/

    for (ia = 0; ia < 3; ia++) {
      f[ia] *= calib_.ftarget;
      calib_.fbar[ia] = 0.0;
      calib_.ubar[ia] = 0.0;
    }
    calib_.ndata = 0;

    physics_fgrav_set(f);

    info("\n\n");
    info("Calibration information:\n");
    info("Target Reynolds number:    %11.4e\n", TARGET_REYNOLDS_NUMBER);
    info("Target particle speed:     %11.4e\n", calib_.utarget);
    info("Force applied:             %11.4e\n", calib_.ftarget);
    info("Spin-up T_diffusion:       %11d\n", calib_.nstart);
    info("Stokes time (timesteps):   %11d\n", calib_.nstokes);
    info("Measurement frequency:     %11d\n", calib_.nfreq);
    info("\n\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_calibration_accumulate
 *
 *  All arguments may be NULL if calibration is not active.
 *
 *****************************************************************************/

int stats_calibration_accumulate(colloids_info_t * cinfo, int ntime,
				 hydro_t * hydro, map_t * map) {

  if (cinfo == NULL) return 0;
  if (hydro == NULL) return 0;

  if (ntime >= calib_.nstart) {
    if ((ntime % calib_.nfreq) == 0) {
      ++calib_.ndata;
      stats_calibration_measure(cinfo, hydro, map);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_calibration_finish
 *
 *  We require at this point for root to have the appropriate
 *  information for output.
 *
 *****************************************************************************/

int stats_calibration_finish(void) {

  int ia;
  double eta;
  double t;
  double ah, ahm1;
  double length;
  double fhasimoto;
  double f0, u0;
  double f[3];
  double u[3];
  double fbar[3];

  if (calib_.nstart < INT_MAX) {

    physics_eta_shear(&eta);
    t = 1.0*calib_.ndata*calib_.nfreq/calib_.nstokes;

    info("\n\n");
    info("Calibration result\n");
    info("Number of measurements:    %11d\n", calib_.ndata);
    info("Run time (Stokes times):   %11.4e\n", t);

    if (calib_.ndata < 1) fatal("No data in stats_calibration_finish\n");

    /* We need to do a reduction on fbar to get the total before
     * taking the average. */

    MPI_Reduce(calib_.fbar, fbar, 3, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

    for (ia = 0; ia < 3; ia++) {
      u[ia] = calib_.ubar[ia]/calib_.ndata;
      f[ia] = fbar[ia]/calib_.ndata;
    }

    /* There is no closed expression for ah in the finite system, so
     * we use the Hasimoto expression to iterate to a solution. Two
     * or three iterations is usually enough for 3 figures of accuracy. */

    length = L(X);
    f0 = modulus(f);
    u0 = modulus(u);

    ah = calib_.a0;

    for (ia = 0; ia < 10; ia++) {
      ahm1 = ah;
      fhasimoto = stats_calibration_hasimoto(ahm1, length);
      ah = 1.0/(6.0*pi_*eta*u0/f0 - (fhasimoto - 1.0)/ahm1);
    }

    fhasimoto = stats_calibration_hasimoto(ah, length);
  
    info("\n");
    info("Actual force:              %11.4e\n", f0);
    info("Actual speed:              %11.4e\n", u0);
    info("Hasimoto correction (a/L): %11.4e\n", fhasimoto);
    info("Input radius:              %11.4e\n", calib_.a0);
    info("Hydrodynamic radius:       %11.4e\n", ah);
    info("Stokes equation rhs:       %11.4e\n", 6.0*pi_*eta*ah*u0);
    info("Stokes equation lhs:       %11.4e\n", f0*fhasimoto);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_calibration_hasimoto
 *
 *  The finite size correction to the Stokes' relation is
 *
 *  6\pi\eta a = (F/U)*[1 - 2.837(a/L) + 4.19(a/L)^3 - 27.4(a/L)^6]
 *
 *  This function returns the fraction [] as a function of a, L.
 *
 *****************************************************************************/

static double stats_calibration_hasimoto(double a, double len) {

  double f;

  f = 1.0 - 2.837*(a/len) + 4.19*pow(a/len, 3) - 27.4*pow(a/len, 6);

  return f;
}

/*****************************************************************************
 *
 *  stats_calibration_measure
 *
 *  Accumulate a measurement of
 *    1. the hydrodynamic force on the particle and
 *    2. the velocity of the particle relative to the fluid.
 *
 *  This is done via an MPI_Reduce to get all the appropriate
 *  data to root, as we don't know where the single particle
 *  is in advance.
 *
 *****************************************************************************/

static int stats_calibration_measure(colloids_info_t * cinfo,
				     hydro_t * hydro, map_t * map) {
  int ic, jc, kc, ia, index;
  int nlocal[3];
  int ncell[3];
  int status;

  double volume;
  double u[3];
  double upart[3];
  double ulocal[3];
  double datalocal[7], datasum[7];
  colloid_t * pc;

  assert(hydro);
  assert(map);

  volume = 0.0;
  for (ia = 0; ia < 3; ia++) {
    upart[ia] = 0.0;
    ulocal[ia] = 0.0;
  }

  /* Find the particle, and record the force and velocity. */

  assert(cinfo);
  colloids_info_ncell(cinfo, ncell);

  for (ic = 1; ic <= ncell[X]; ic++) {
    for (jc = 1; jc <= ncell[Y]; jc++) {
      for (kc = 1; kc <= ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	if (pc) {
	  for (ia = 0; ia < 3; ia++) {
	    calib_.fbar[ia] += pc->force[ia];
	    upart[ia] = pc->s.v[ia];
	  }
	  break;
	}
      }
    }
  }

  /* Work out the fluid velocity */

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	hydro_u(hydro, index, u);
	ulocal[X] += u[X];
	ulocal[Y] += u[Y];
	ulocal[Z] += u[Z];
	volume = volume + 1.0;
      }
    }
  }

  /* Sum these data to root. */

  datalocal[0] = upart[X];
  datalocal[1] = upart[Y];
  datalocal[2] = upart[Z];
  datalocal[3] = ulocal[X];
  datalocal[4] = ulocal[Y];
  datalocal[5] = ulocal[Z];
  datalocal[6] = volume;

  MPI_Reduce(datalocal, datasum, 7, MPI_DOUBLE, MPI_SUM, 0, pe_comm());

  /* Only root gets this right. That's ok for output. */

  for (ia = 0; ia < 3; ia++) {
    upart[ia] = datasum[ia];
    ulocal[ia] = datasum[3+ia]/datasum[6];
    calib_.ubar[ia] += (upart[ia] - ulocal[ia]);
  }

  return 0;
}
