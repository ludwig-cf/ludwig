/*****************************************************************************
 *
 *  Ludwig
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *****************************************************************************/

#include <stdio.h>

#include "pe.h"
#include "runtime.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
#include "control.h"
#include "free_energy.h"
#include "model.h"
#include "bbl.h"
#include "subgrid.h"

#include "colloids.h"
#include "collision.h"
#include "test.h"
#include "wall.h"
#include "communicate.h"
#include "leesedwards.h"
#include "interaction.h"
#include "propagation.h"

#include "lattice.h"
#include "cio.h"
#include "regsteer.h"

static char rcsid[] = "$Id: main.c,v 1.12 2007-12-05 17:38:12 kevin Exp $";

int main( int argc, char **argv )
{
  char    filename[256];
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

  while (next_step()) {

    TIMER_start(TIMER_STEPS);
    step = get_step();

    latt_zero_force();
    COLL_update();
    wall_update();

    /* Collision stage */
    collide();

    LE_apply_LEBC();
    halo_site();

    /* Colloid bounce-back applied between collision and
     * propagation steps. */

#ifdef _SUBGRID_
    subgrid_update();
#else
    bounce_back_on_links();
    wall_bounce_back();
#endif

    /* There must be no halo updates between bounce back
     * and propagation, as the halo regions hold active f,g */

    propagation();

    TIMER_stop(TIMER_STEPS);

    /* Configuration dump */

    if (is_config_step()) {
      COM_write_site(get_output_config_filename(step), MODEL_write_site);
      sprintf(filename, "%s%6.6d", "config.cds", step);
      CIO_write_state(filename);
    }

    /* Measurements */


    if (is_measurement_step()) {	  
      info("Wrting phi file at  at step %d!\n", step);
      /*COLL_compute_phi_missing();*/
      sprintf(filename,"phi-%6.6d",step);
      COM_write_site(filename, MODEL_write_phi);
      sprintf(filename, "%s%6.6d", "config.cds", step);
      CIO_write_state(filename);
    }

    /* Print progress report */

    if (is_statistics_step()) {

      MISC_curvature();
      TEST_statistics();
      TEST_momentum();
#ifdef _NOISE_
      TEST_fluid_temperature();
#endif

      info("\nCompleted cycle %d\n", step);
    }

    /* Next time step */
  }


  /* Dump the final configuration if required. */

  if (is_config_at_end()) {
    COM_write_site(get_output_config_filename(step), MODEL_write_site);
    sprintf(filename, "%s%6.6d", "config.cds", step);
    CIO_write_state(filename);
  }

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
