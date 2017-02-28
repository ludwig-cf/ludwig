/*****************************************************************************
 *
 *  test_timer
 *
 *  Test the timing routines.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <time.h>
#include <limits.h>

#include "pe.h"
#include "timer.h"
#include "tests.h"

/*****************************************************************************
 *
 *  test_timer_suite
 *
 *****************************************************************************/

int test_timer_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  TIMER_init(pe);
  TIMER_start(TIMER_TOTAL);
  TIMER_stop(TIMER_TOTAL);

  pe_info(pe, "PASS     ./unit/test_timer\n");
  pe_free(pe);

  return 0;
}
