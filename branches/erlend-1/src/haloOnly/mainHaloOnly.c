/*****************************************************************************
 *
 *  Ludwig
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *****************************************************************************/

#include <stdio.h>

#include "../pe.h"
#include "../runtime.h"
#include "../ran.h"
#include "../timer.h"
#include "../coords.h"
#include "../control.h"
#include "../free_energy.h"
#include "../model.h"
#include "../bbl.h"
#include "../subgrid.h"

#include "../colloids.h"
#include "../collision.h"
#include "../test.h"
#include "../wall.h"
#include "../communicate.h"
#include "../leesedwards.h"
#include "../interaction.h"
#include "../propagation.h"

#include "../lattice.h"
#include "../cio.h"
#include "../regsteer.h"


static char rcsid[] = "$Id: mainHaloOnly.c,v 1.1.2.1 2008-02-06 17:55:44 erlend Exp $";

int main( int argc, char **argv )
{
  char    filename[FILENAME_MAX];
  int     step;

  /* Initialise the following:
   *    - RealityGrid steering (if required)
   *    - communications (MPI)
   *    - random number generation (serial RNG and parallel fluctuations)
   *    - model fields
   *    - simple walls 
   *    - colloidal particles */

  REGS_init();

  pe_init(argc, argv);
  if (argc > 1) {
    RUN_read_input_file(argv[1]);
  }
  else {
    RUN_read_input_file("input");
  }
  coords_init();
  init_control();

  COM_init();

  TIMER_init();
  TIMER_start(TIMER_TOTAL);

  ran_init();
  RAND_init_fluctuations();
  MODEL_init();
  LE_init();
  wall_init();
  COLL_init();

  init_free_energy();

  /* Report initial statistics */

  TEST_statistics();
  TEST_momentum();

  /* Main time stepping loop */
  TIMER_start(TIMER_STEPS);
  for(step=0; step<10000; step++) {
	halo_site();
  }
  TIMER_stop(TIMER_STEPS);

  /* Shut down cleanly. Give the timer statistics. Finalise PE. */

  COLL_finish();
  wall_finish();

  TIMER_stop(TIMER_TOTAL);
  TIMER_statistics();

  pe_finalise();
  REGS_finish();

  return 0;
}

/*****************************************************************************
 *
 *  print_shear_profile
 *
 *****************************************************************************/

void print_shear_profile() {

  int index;
  int ic, jc = 1, kc = 1;
  int N[ND];
  double rho, u[ND];

  info("Shear profile\n\n");
  get_N_local(N);

  for (ic = 1; ic <= N[X]; ic++) {

    index = index_site(ic, jc, kc);
    rho = get_rho_at_site(index);
    get_momentum_at_site(index, u);

    printf("%4d %10.8f %10.8f\n", ic, rho, u[Y]/rho);
  }

  return;
}
