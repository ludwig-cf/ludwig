/*****************************************************************************
 *
 *  test_timer
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <time.h>
#include <limits.h>

#include "pe.h"
#include "timer.h"
#include "test.h"

int main(int argc, char ** argv) {

  int i;
  double t;

  pe_init(argc, argv);

  info("\nTesting timer routines...\n");

  info("sizeof(clock_t) is %d bytes\n", sizeof(clock_t));
  info("CLOCKS_PER_SEC is %d\n", CLOCKS_PER_SEC);
  info("LONG_MAX is %ld\n", LONG_MAX);
  info("Estimate maximum serial realaible times %f seconds\n",
       ((double) (LONG_MAX))/CLOCKS_PER_SEC);

  TIMER_init();
  TIMER_start(TIMER_TOTAL);
  TIMER_stop(TIMER_TOTAL);
  TIMER_statistics();

  pe_finalise();

  return 0;
}
