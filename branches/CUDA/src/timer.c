/*****************************************************************************
 *
 *  timer.c
 *
 *  Performance timer routines.
 *
 *  There are a number of separate 'timers', each of which can
 *  be started, and stopped, independently.
 *
 *  $Id: timer.c,v 1.5 2010/10/15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <time.h>
#include <float.h>

#include "pe.h"
#include "util.h"
#include "timer.h"

#define NTIMERS  50

struct timer_struct {
  double          t_last;
  double          t_start;
  double          t_sum;
  double          t_max;
  double          t_min;
  unsigned int    active;
  unsigned int    nsteps;
};

static struct timer_struct timer[NTIMERS];

static const char * timer_name[] = {"Total",
				    "Time step loop",
				    "Propagation",
				    "Collision",
				    "Collision Wait",
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
				    "Ewald Sum total",
				    "Ewald Real",
				    "Ewald_Fourier",
				    "Force calculation",
				    "phi update",
				    "Free1",
				    "Free2",
                                    "Free3",
				    "Phi Comp",
				    "Phi d->h",
				    "Phi Halo",
				    "Phi Grad Comp",
				    "Force h->d",
				    "Phi h->d",
				    "Velocity d->h",
				    "Edge pack d",
				    "Edge d->h",
				    "Phi Edge pack d",
				    "Phi Edge d->h",
				    "Phi Update Malloc",
				    "Phi Update Upwind",
				    "Phi Update Advec",
				    "Phi Update BE",
				    "Halo CUDA Put Y",
				    "Halo CUDA Put Z",
				    "Halo MPI X",
				    "Halo MPI Y",
				    "Halo MPI Z",
				    "Halo Y Corner",
				    "Halo Z Corner",
				    "Phi Halo h->d",
				    "Phi Halo unpack d",
				    "Velocity Halo"};


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

  timer[t_id].t_start = MPI_Wtime();
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

    t_elapse = MPI_Wtime() - timer[t_id].t_start;

    timer[t_id].t_last = t_elapse;
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

  MPI_Comm comm = pe_comm();

  r = MPI_Wtick();

  info("\nTimer resolution: %g second\n", r);
  //  info("\nTimer statistics **ON MPI ROOT TASK**\n");
  info("\nTimer statistics\n");
  info("%20s: %10s %10s %10s\n", "Section", "  tmin", "  tmax", " total");

  for (n = 0; n < NTIMERS; n++) {

    /* Report the stats for active timers */
        
    if (timer[n].nsteps != 0) {

      
      t_min = timer[n].t_min;
      t_max = timer[n].t_max;
      t_sum = timer[n].t_sum;

      
      MPI_Reduce(&(timer[n].t_min), &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(&(timer[n].t_max), &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      MPI_Reduce(&(timer[n].t_sum), &t_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

      t_sum /= pe_size();

      info("%20s: %10.3f %10.3f %10.3f %10.6f", timer_name[n],
	   t_min, t_max, t_sum, t_sum/(double) timer[n].nsteps);
      info(" (%d call%s)\n", timer[n].nsteps, timer[n].nsteps > 1 ? "s" : ""); 
    }
  }

  return;
}


/* print last recorded timings on MPI root task */
void TIMER_statistics_last() {

  int    n;
  double r;

  MPI_Comm comm = pe_comm();

  r = MPI_Wtick();

  info("\nTimer resolution: %g second\n", r);
  info("\nTimer statistics **ON MPI ROOT TASK**\n");
  info("%20s: %10s %10s\n", "Section", "  tlast", "step");

  for (n = 0; n < NTIMERS; n++) {

    /* Report the stats for active timers */
        
    if (timer[n].nsteps != 0) {
      info("%20s: %10.5f %d\n", timer_name[n],timer[n].t_last,get_step());

    }
  }

  return;
}
