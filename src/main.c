/*****************************************************************************
 *
 *  Ludwig
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *****************************************************************************/

#include "pe.h"
#include "runtime.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
#include "cartesian.h"
#include "control.h"
#include "free_energy.h"

#include "globals.h"
#include "cmem.h"
#include "cio.h"
#include "regsteer.h"

static char rcsid[] = "$Id: main.c,v 1.5 2006-08-03 11:27:26 kevin Exp $";


int main( int argc, char **argv )
{
  char    filename[256];
  int     step;

  /* Initialise the following:
   *    - RealityGrid steering (if required)
   *    - communications (MPI)
   *    - random number generation (serial RNG and parallel fluctuations)
   *    - model fields
   *    - utilities
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
  cart_init();
  init_control();

  COM_init( argc, argv );

  TIMER_init();
  TIMER_start(TIMER_TOTAL);

  ran_init();
  RAND_init_fluctuations();
  MODEL_init();
  WALL_init();
  COLL_init();

  init_free_energy();

  /* Report initial statistics */

  TEST_statistics();
  TEST_momentum();


  /* Main time stepping loop */

  while (next_step()) {

    step = get_step();
    TIMER_start(TIMER_STEPS);

#ifdef _REGS_
    {
      int stop;
      stop = REGS_control(step);
      if (stop) break;
    }
#endif

#ifdef _COLLOIDS_

    COM_halo();
    COLL_update();
    COLL_forces();

    MODEL_collide_multirelaxation();
    COM_halo();

    /* Colloid bounce-back applied between collision and
     * propagation steps. */

    COLL_bounce_back();

    /* There must be no halo updates between COLL_bounce_back
     *  and propagation, as the halo regions hold active f,g */

    propagation();

#else
    /* No colloids */

    MODEL_collide_multirelaxation();

    LE_apply_LEBC();

    COM_halo();
    propagation();
#endif /* _COLLOIDS_ */

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
      TIMER_start(TIMER_IO);
      sprintf(filename, "%s%6.6d", "config.cds", step);
      CIO_write_state(filename);
      TIMER_stop(TIMER_IO);
    }

    /* Print progress report */

    if (is_statistics_step()) {

      MISC_curvature();
      TEST_statistics();
      TEST_momentum();
#ifdef _NOISE_
      TEST_fluid_temperature();
#endif
#ifdef _COLLOIDS_
      CMEM_report_memory();
#endif
      info("\nCompleted cycle %d\n", step);
    }

    /* Next time step */
  }

  /* Finalise the following:
   *   - dump the final configuration
   *   - LE parameters (required?)
   *   - dump final colloid configuration and dellocate
   *   - close down model
   *   - terminate communications */


  COM_write_site(get_output_config_filename(step), MODEL_write_site);
  LE_print_params();
  sprintf(filename, "%s%6.6d", "config.cds", step);
  CIO_write_state(filename);
  COLL_finish();

  TIMER_stop(TIMER_TOTAL);
  TIMER_statistics();

  pe_finalise();
  REGS_finish();

  return 0;
}
