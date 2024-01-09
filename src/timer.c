/*****************************************************************************
 *
 *  timer.c
 *
 *  Performance timer routines.
 *
 *  There are a number of separate 'timers', each of which can
 *  be started, and stopped, independently.
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
#include <time.h>
#include <float.h>

#include "pe.h"
#include "util.h"
#include "timer.h"

struct timer_struct {
  double          t_start;
  double          t_sum;
  double          t_max;
  double          t_min;
  unsigned int    active;
  unsigned int    nsteps;
};

static pe_t * pe_stat = NULL;
static struct timer_struct timer[TIMER_NTIMERS];

static const char * timer_name[] = {"Total",
				    "Time step loop",
				    "Propagation",
				    "Propagtn (krnl) ",
				    "Collision",
				    "Collision (krnl) ",
				    "Lattice halos",
				    "-> imbalance",
				    "-> irecv",
				    "-> pack",
				    "-> isend",
				    "-> waitall",
				    "-> unpack",
				    "phi gradients",
				    "phi grad (krnl) ",
				    "phi halos",
				    "-> imbalance",
				    "-> irecv",
				    "-> pack",
				    "-> isend",
				    "-> waitall",
				    "-> unpack",
				    "Lees Edwards BC",
				    "I/O",
				    "Forces",
				    "Rebuild",
				    "BBL",
				    "Particle updates",
				    "Particle halos",
				    "Fluctuations",
				    "Ewald Sum total",
				    "Ewald Real",
				    "Ewald_Fourier",
				    "Force calculation",
				    "Chem stress (krnl) ",
				    "Phi force (krnl) ",
				    "phi update",
				    "Velocity Halo ",
				    "-> imbalance",
				    "-> irecv",
				    "-> pack",
				    "-> isend",
				    "-> waitall",
				    "-> unpack",
				    "BE mol field (krnl) ",
				    "BP BE update (krnl) ",
				    "Advectn (krnl) ",
				    "Advectn BCS (krnl) ",
				    "BP BE alloc/free ",
				    "Electrokinetics",
				    "Poisson equation",
				    "Nernst Planck",
				    "Lap timer (no report)",
				    "Diagnostics / output",
				    "Free2",
                                    "Free3", "Free4", "Free5", "Free6"
};


/****************************************************************************
 *
 *  TIMER_init
 *
 *  Make sure everything is set to go.
 *
 ****************************************************************************/

int TIMER_init(pe_t * pe) {

  int n;

  pe_stat = pe;

  for (n = 0; n < TIMER_NTIMERS; n++) {
    timer[n].t_sum  = 0.0;
    timer[n].t_max  = FLT_MIN;
    timer[n].t_min  = FLT_MAX;
    timer[n].active = 0;
    timer[n].nsteps = 0;
  }

  return 0;
}


/****************************************************************************
 *
 *  TIMER_start
 *
 *  Start timer for the specified timer.
 *
 ****************************************************************************/

void TIMER_start(const int t_id) {

  timer[t_id].t_start = MPI_Wtime();
  timer[t_id].active  = 1;
  timer[t_id].nsteps += 1;

  return;
}

/*****************************************************************************
 *
 *  timer_lapse
 *
 *  Return lapsed time (seconds) since previous call (or start).
 *
 *****************************************************************************/

double timer_lapse(const int id) {

  double tlapse = -999.999;

  assert(timer[id].active);

  {
    double tnow = MPI_Wtime();

    tlapse = tnow - timer[id].t_start;
    timer[id].t_start = tnow;
    timer[id].nsteps  = 0;
  }

  return tlapse;
}

/****************************************************************************
 *
 *  TIMER_stop_timer
 *
 *  Stop the specified timer and add the elapsed time to the total.
 *
 ****************************************************************************/

void TIMER_stop(const int t_id) {

  double t_elapse;

  if (timer[t_id].active) {

    t_elapse = MPI_Wtime() - timer[t_id].t_start;

    timer[t_id].t_sum += t_elapse;
    timer[t_id].t_max  = dmax(timer[t_id].t_max, t_elapse);
    timer[t_id].t_min  = dmin(timer[t_id].t_min, t_elapse);
    timer[t_id].active = 0;
  }

  return;
}

/*****************************************************************************
 *
 *  TIMER_statistics
 *
 *  Print a digestable overview of the time statistics.
 *
 *****************************************************************************/

void TIMER_statistics() {

  int    n;
  double t_min, t_max, t_sum;
  double r;

  MPI_Comm comm;

  assert(pe_stat);
  r = MPI_Wtick();

  pe_mpi_comm(pe_stat, &comm);
  pe_info(pe_stat, "\nTimer resolution: %g second\n", r);
  pe_info(pe_stat, "\nTimer statistics\n");
  pe_info(pe_stat, "%20s: %10s %10s %10s\n", "Section", "  tmin", "  tmax", " total");

  for (n = 0; n < TIMER_NTIMERS; n++) {

    /* Report the stats for active timers */
    /* Not the lap timer. */

    if (n == TIMER_LAP) continue;
    
    if (timer[n].nsteps != 0) {

      t_min = timer[n].t_min;
      t_max = timer[n].t_max;
      t_sum = timer[n].t_sum;

      MPI_Reduce(&(timer[n].t_min), &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(&(timer[n].t_max), &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      MPI_Reduce(&(timer[n].t_sum), &t_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

      t_sum /= pe_mpi_size(pe_stat);

      pe_info(pe_stat, "%20s: %10.3f %10.3f %10.3f %10.6f", timer_name[n],
	   t_min, t_max, t_sum, t_sum/(double) timer[n].nsteps);
      pe_info(pe_stat, " (%d call%s)\n", timer[n].nsteps, timer[n].nsteps > 1 ? "s" : ""); 
    }
  }

  return;
}

/*****************************************************************************
 *
 *  timekeeper_create
 *
 *****************************************************************************/

__host__ int timekeeper_create(pe_t * pe, const timekeeper_options_t * opts,
			       timekeeper_t * tk) {
  assert(pe);
  assert(opts);
  assert(tk);

  *tk = (timekeeper_t) {0};
  tk->pe = pe;
  tk->options = *opts;

  return 0;
}

/*****************************************************************************
 *
 *  timekeeper_step
 *
 *  We anticipate calling this once at end of each time step ..
 *
 *****************************************************************************/

__host__ int timekeeper_step(timekeeper_t * tk) {

  assert(tk);

  tk->timestep += 1;

  if (tk->options.lap_report) {
    if (tk->timestep % tk->options.lap_report_freq == 0) {
      /* Recell strctime from pe_time() has a new line */
      pe_t * pe = tk->pe;
      char strctime[BUFSIZ] = {0};
      pe_time(strctime, BUFSIZ);
      pe_info(pe, "\nLap time at step %9d is: %8.3f seconds at %s",
	      tk->timestep, timer_lapse(TIMER_LAP), strctime);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  timekeeper_free
 *
 *****************************************************************************/

__host__ int timekeeper_free(timekeeper_t * tk) {

  assert(tk);

  *tk = (timekeeper_t) {0};

  return 0;
}
