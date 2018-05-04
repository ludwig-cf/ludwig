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
 *  (c) 2011-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <math.h>

#include "ran.h"
#include "util.h"
#include "physics.h"
#include "stats_calibration.h"

#define TARGET_REYNOLDS_NUMBER 0.05
#define MEASUREMENTS_PER_STOKES_TIME 50

struct stats_ahydro_s {
  pe_t * pe;
  cs_t * cs;
  map_t * map;
  hydro_t * hydro;
  colloids_info_t * cinfo;
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

static double stats_calibration_hasimoto(double a, double length);
static int stats_ahydro_measure(stats_ahydro_t * stat);

/*****************************************************************************
 *
 *  stats_calibration_create
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

int stats_ahydro_create(pe_t * pe, cs_t * cs, colloids_info_t * cinfo,
			hydro_t * hydro, map_t * map, stats_ahydro_t ** pobj) {

  int ia;
  int nctotal;
  int ntotal[3];
  double a;
  double rho;
  double eta;
  double length;
  double fhasimoto;
  double f[3];
  physics_t * phys = NULL;
  PI_DOUBLE(pi);
  stats_ahydro_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(cinfo);
  assert(hydro);
  assert(map);
  assert(pobj);

  obj = (stats_ahydro_t *) calloc(1, sizeof(stats_ahydro_t));
  if (obj == NULL) pe_fatal(pe, "calloc(stats_ahydro_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->map = map;
  obj->cinfo = cinfo;
  obj->hydro = hydro;

  cs_ntotal(cs, ntotal);
  physics_ref(&phys);

  /* Make sure we have a cubic system with one particle */

  assert(ntotal[X] == ntotal[Y]);
  assert(ntotal[Y] == ntotal[Z]);
  colloids_info_ntotal(cinfo, &nctotal);
  if (nctotal != 1) pe_fatal(pe, "Calibration requires exactly one colloid\n");

  length = 1.0*ntotal[Z];
  physics_rho0(phys, &rho);
  physics_eta_shear(phys, &eta);

  colloids_info_ahmax(cinfo, &a);

  obj->a0 = a;
  obj->utarget = eta*TARGET_REYNOLDS_NUMBER/(a*rho);
  fhasimoto = stats_calibration_hasimoto(a, length);
  obj->ftarget = 6.0*pi*eta*a*obj->utarget/fhasimoto;

  obj->nstokes = a/obj->utarget;
  obj->nfreq = obj->nstokes/MEASUREMENTS_PER_STOKES_TIME;
  if (obj->nfreq < 1) obj->nfreq = 1;
  obj->nstart = length*length/eta;

  /* Set a force of the right size in a random direction, and zero
   * the accumulators. The actual numbers come from an old RNG and
   * are retained as literals to allow tests to pass. */

  f[X] = +5.02274083742018e-01;
  f[Y] = -1.05061333197473e-01;
  f[Z] = -8.58302313330149e-01;

  for (ia = 0; ia < 3; ia++) {
    f[ia] *= obj->ftarget;
    obj->fbar[ia] = 0.0;
    obj->ubar[ia] = 0.0;
  }
  obj->ndata = 0;

  physics_fgrav_set(phys, f);

  pe_info(pe, "\n\n");
  pe_info(pe, "Calibration information:\n");
  pe_info(pe, "Target Reynolds number:    %11.4e\n", TARGET_REYNOLDS_NUMBER);
  pe_info(pe, "Target particle speed:     %11.4e\n", obj->utarget);
  pe_info(pe, "Force applied:             %11.4e\n", obj->ftarget);
  pe_info(pe, "Spin-up T_diffusion:       %11d\n", obj->nstart);
  pe_info(pe, "Stokes time (timesteps):   %11d\n", obj->nstokes);
  pe_info(pe, "Measurement frequency:     %11d\n", obj->nfreq);
  pe_info(pe, "\n\n");

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  stats_ahydro_accumulate
 *
 *  All arguments may be NULL if calibration is not active.
 *
 *****************************************************************************/

int stats_ahydro_accumulate(stats_ahydro_t * stat, int ntime) {

  if (stat == NULL) return 0;

  if (ntime >= stat->nstart) {
    if ((ntime % stat->nfreq) == 0) {
      ++stat->ndata;
      hydro_memcpy(stat->hydro, tdpMemcpyDeviceToHost);
      map_memcpy(stat->map, tdpMemcpyDeviceToHost);
      stats_ahydro_measure(stat);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stats_ahydro_free
 *
 *  We require at this point for root to have the appropriate
 *  information for output.
 *
 *****************************************************************************/

int stats_ahydro_free(stats_ahydro_t * stat) {

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
  double ltot[3];
  physics_t * phys = NULL;
  PI_DOUBLE(pi);
  MPI_Comm comm;

  assert(stat);

  if (stat->nstart < INT_MAX) {

    physics_ref(&phys);
    physics_eta_shear(phys, &eta);
    t = 1.0*stat->ndata*stat->nfreq/stat->nstokes;

    pe_info(stat->pe, "\n\n");
    pe_info(stat->pe, "Calibration result\n");
    pe_info(stat->pe, "Number of measurements:    %11d\n", stat->ndata);
    pe_info(stat->pe, "Run time (Stokes times):   %11.4e\n", t);

    if (stat->ndata < 1) pe_fatal(stat->pe, "No data in stats_ahydro_free\n");

    /* We need to do a reduction on fbar to get the total before
     * taking the average. */

    pe_mpi_comm(stat->pe, &comm);
    MPI_Reduce(stat->fbar, fbar, 3, MPI_DOUBLE, MPI_SUM, 0, comm);

    for (ia = 0; ia < 3; ia++) {
      u[ia] = stat->ubar[ia]/stat->ndata;
      f[ia] = fbar[ia]/stat->ndata;
    }

    /* There is no closed expression for ah in the finite system, so
     * we use the Hasimoto expression to iterate to a solution. Two
     * or three iterations is usually enough for 3 figures of accuracy. */

    cs_ltot(stat->cs, ltot);
    length = ltot[X];
    f0 = modulus(f);
    u0 = modulus(u);

    ah = stat->a0;

    for (ia = 0; ia < 10; ia++) {
      ahm1 = ah;
      fhasimoto = stats_calibration_hasimoto(ahm1, length);
      ah = 1.0/(6.0*pi*eta*u0/f0 - (fhasimoto - 1.0)/ahm1);
    }

    fhasimoto = stats_calibration_hasimoto(ah, length);
  
    pe_info(stat->pe, "\n");
    pe_info(stat->pe, "Actual force:              %11.4e\n", f0);
    pe_info(stat->pe, "Actual speed:              %11.4e\n", u0);
    pe_info(stat->pe, "Hasimoto correction (a/L): %11.4e\n", fhasimoto);
    pe_info(stat->pe, "Input radius:              %11.4e\n", stat->a0);
    pe_info(stat->pe, "Hydrodynamic radius:       %11.4e\n", ah);
    pe_info(stat->pe, "Stokes equation rhs:       %11.4e\n", 6.0*pi*eta*ah*u0);
    pe_info(stat->pe, "Stokes equation lhs:       %11.4e\n", f0*fhasimoto);
  }

  free(stat);

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
 *  stats_ahydro_measure
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

static int stats_ahydro_measure(stats_ahydro_t * stat) {

  int ic, jc, kc, ia, index;
  int nlocal[3];
  int status;

  double volume;
  double u[3];
  double upart[3];
  double ulocal[3];
  double datalocal[7], datasum[7];
  colloid_t * pc;
  MPI_Comm comm;

  assert(stat);

  volume = 0.0;
  for (ia = 0; ia < 3; ia++) {
    upart[ia] = 0.0;
    ulocal[ia] = 0.0;
  }

  /* Find the particle, and record the force and velocity. */

  colloids_info_local_head(stat->cinfo, &pc);

  if (pc) {
    for (ia = 0; ia < 3; ia++) {
      stat->fbar[ia] += pc->force[ia];
      upart[ia] = pc->s.v[ia];
    }
  }

  /* Work out the fluid velocity */

  cs_nlocal(stat->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(stat->cs, ic, jc, kc);
	map_status(stat->map, index, &status);
	if (status != MAP_FLUID) continue;

	hydro_u(stat->hydro, index, u);
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

  pe_mpi_comm(stat->pe, &comm);
  MPI_Reduce(datalocal, datasum, 7, MPI_DOUBLE, MPI_SUM, 0, comm);

  /* Only root gets this right. That's ok for output. */

  for (ia = 0; ia < 3; ia++) {
    upart[ia] = datasum[ia];
    ulocal[ia] = datasum[3+ia]/datasum[6];
    stat->ubar[ia] += (upart[ia] - ulocal[ia]);
  }

  return 0;
}
