/*****************************************************************************
 *
 *  timer.c
 *
 *  Performance timer routines.
 *
 *  There are a number of separate 'timers', each of which can
 *  be started, and stopped, independently.
 *
 *****************************************************************************/

#include <time.h>
#include <float.h>

#include "pe.h"
#include "timer.h"

#define NTIMERS  16

struct timer_struct {
  double          t_start;
  double          t_sum;
  double          t_max;
  double          t_min;
  unsigned int    active;
  unsigned int    nsteps;
};

static struct timer_struct timer[NTIMERS];
static double TIMER_get_seconds(void);

static const char * timer_name[] = {"Total",
				    "Time step loop",
				    "Propagation",
				    "Collision",
				    "Lattice halos",
				    "phi gradients",
				    "Lees Edwards BC",
				    "I/O",
				    "Forces",
				    "Rebuild",
				    "BBL",
				    "Particle updates",
				    "Particle halos",
				    "Fluctuations",
				    "Free1",
				    "Free2",
                                    "Free3"};


/****************************************************************************
 *
 *  TIMER_init
 *
 *  Make sure everything is set to go.
 *
 ****************************************************************************/

void TIMER_init() {

  int n;

  for (n = 0; n < NTIMERS; n++) {
    timer[n].t_sum  = 0.0;
    timer[n].t_max  = FLT_MIN;
    timer[n].t_min  = FLT_MAX;
    timer[n].active = 0;
    timer[n].nsteps = 0;
  }

  return;
}


/****************************************************************************
 *
 *  TIMER_start
 *
 *  Start timer for the specified timer.
 *
 ****************************************************************************/

void TIMER_start(const int t_id) {

  timer[t_id].t_start = TIMER_get_seconds();
  timer[t_id].active  = 1;
  timer[t_id].nsteps += 1;

  return;
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

    t_elapse = TIMER_get_seconds() - timer[t_id].t_start;

    timer[t_id].t_sum += t_elapse;
    timer[t_id].t_max  = dmax(timer[t_id].t_max, t_elapse);
    timer[t_id].t_min  = dmin(timer[t_id].t_min, t_elapse);
    timer[t_id].active = 0;
  }

  return;
}


/*****************************************************************************
 *
 *  TIMER_get_seconds
 *
 *  This should return the time since some fixed point in
 *  the past in seconds.
 *
 *****************************************************************************/

double TIMER_get_seconds() {

#ifdef _MPI_
  return MPI_Wtime();
#else
  return ((double) clock()) / CLOCKS_PER_SEC;
#endif

}


/*****************************************************************************
 *
 *  TIMER_statistics
 *
 *  Print a digestable overview of the time statistics.
 *  Communication is assumed to take place within MPI_COMM_WORLD.
 *
 *****************************************************************************/

void TIMER_statistics() {

  int    n;
  double t_min, t_max, t_sum;

  info("\nTimer statistics\n");
  info("%20s: %10s %10s %10s\n", "Section", "  tmin", "  tmax", " total");

  for (n = 0; n < NTIMERS; n++) {

    /* Report the stats for active timers */

    if (timer[n].nsteps != 0) {

      t_min = timer[n].t_min;
      t_max = timer[n].t_max;
      t_sum = timer[n].t_sum;

#ifdef _MPI_
      MPI_Reduce(&(timer[n].t_min), &t_min, 1, MPI_DOUBLE, MPI_MIN, 0,
		 MPI_COMM_WORLD);
      MPI_Reduce(&(timer[n].t_max), &t_max, 1, MPI_DOUBLE, MPI_MAX, 0,
		 MPI_COMM_WORLD);
      MPI_Reduce(&(timer[n].t_sum), &t_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
		   MPI_COMM_WORLD);

      t_sum /= pe_size();
#endif

      info("%20s: %10.3f %10.3f %10.3f %10.6f", timer_name[n],
	   t_min, t_max, t_sum, t_sum/(double) timer[n].nsteps);
      info(" (%d call%s)\n", timer[n].nsteps, timer[n].nsteps > 1 ? "s" : ""); 
    }
  }

  return;
}
