/*****************************************************************************
 *
 *  Ludwig
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "ran.h"
#include "timer.h"
#include "coords_rt.h"
#include "coords.h"
#include "control.h"
#include "free_energy_rt.h"
#include "model.h"
#include "bbl.h"
#include "subgrid.h"

#include "colloids.h"
#include "collision.h"
#include "wall.h"
#include "communicate.h"
#include "leesedwards.h"
#include "interaction.h"
#include "propagation.h"

#include "site_map.h"
#include "physics.h"
#include "lattice.h"
#include "cio.h"

#include "io_harness.h"
#include "phi.h"
#include "phi_stats.h"
#include "phi_force.h"
#include "phi_force_colloid.h"
#include "phi_gradients.h"
#include "phi_lb_coupler.h"
#include "phi_update.h"
#include "phi_update_rt.h"
#include "blue_phase.h"
#include "model_le.h"
#include "colloid_io_rt.h"
#include "colloids_Q_tensor.h"

#include "advection_rt.h"
#include "distribution_rt.h"
#include "collision_rt.h"
#include "gradient_rt.h"

#include "stats_colloid.h"
#include "stats_turbulent.h"
#include "stats_surfactant.h"
#include "stats_rheology.h"
#include "stats_free_energy.h"
#include "stats_distribution.h"
#include "stats_calibration.h"

#ifdef _GPU_
#include "interface_gpu.h"
#endif

void ludwig_rt(void);
void ludwig_init(void);
void ludwig_report_momentum(void);



/*****************************************************************************
 *
 *  ludwig_rt
 *
 *  Digest the run-time arguments for different parts of the code.
 *
 *****************************************************************************/

void ludwig_rt(void) {

  TIMER_init();
  TIMER_start(TIMER_TOTAL);

  free_energy_run_time();
  phi_update_run_time();
  advection_run_time();

  coords_run_time();

  init_control();

  init_physics();
  le_init();

  distribution_run_time();
  collision_run_time();

  ran_init();
  MODEL_init();
  site_map_init();
  wall_init();
  colloid_io_run_time();
  COLL_init();

  gradient_run_time();

  return;
}

/*****************************************************************************
 *
 *  ludwig_init
 *
 *  Initialise.
 *
 *****************************************************************************/

void ludwig_init(void) {

  int n, nstat;
  char filename[FILENAME_MAX];

  /* Initialise Lc in colloids */
  if (phi_nop() == 5) COLL_randomize_Q(0.0);

  if (get_step() == 0) {
    n = 0;
    RUN_get_int_parameter("LE_init_profile", &n);

    if (n != 0) model_le_init_shear_profile();
  }
  else {
    /* Distributions */

    sprintf(filename, "dist-%8.8d", get_step());
    info("Re-starting simulation at step %d with data read from "
	 "config\nfile(s) %s\n", get_step(), filename);

    io_read(filename, distribution_io_info());

    if (phi_is_finite_difference()) {
      sprintf(filename,"phi-%8.8d", get_step());
      info("Reading phi state from %s\n", filename);
      io_read(filename, io_info_phi);
    }
  }

  phi_gradients_init();

  stats_rheology_init();
  stats_turbulent_init();

  /* Calibration statistics required? */

  nstat = 0;
  n = RUN_get_string_parameter("calibration", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) nstat = 1;
  stats_calibration_init(nstat);

  collision_init();

  return;
}

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main( int argc, char **argv ) {

  char    filename[FILENAME_MAX];
  int     step = 0;

  pe_init(argc, argv);

  if (argc > 1) {
    RUN_read_input_file(argv[1]);
  }
  else {
    RUN_read_input_file("input");
  }

  ludwig_rt();
  ludwig_init();
  
  scalar_q_io_init();

  /* Report initial statistics */

  info("Initial conditions.\n");
  stats_distribution_print();
  phi_stats_print_stats();
  ludwig_report_momentum();

#ifdef _GPU_
  info("Running GPU version\n");
  initialise_gpu();
  put_f_on_gpu();
  put_force_on_gpu(); 
  put_velocity_on_gpu();
 
  /* sync MPI tasks for timing purposes */
  MPI_Barrier(cart_comm());
#endif


  /* Main time stepping loop */

  info("\n");
  info("Starting time step loop.\n");



  while (next_step()) {



    TIMER_start(TIMER_STEPS);

 
    step = get_step();


    hydrodynamics_zero_force();

    COLL_update();
    wall_update();

#ifdef _GPU_
    put_phi_on_gpu();
    put_site_map_on_gpu();
#endif


    /* Collision stage */

    if (phi_nop()) {

      TIMER_start(TIMER_PHI_GRADIENTS);

      /* Note that the liquid crystal boundary conditions must come after
       * the halo swap, but before the gradient calculation. */

#ifdef _GPU_

      TIMER_start(PHICOMP);
      phi_compute_phi_site_gpu();
      TIMER_stop(PHICOMP);

      TIMER_start(PHIHALO);
      phi_halo_swap_gpu();
      TIMER_stop(PHIHALO);


      TIMER_start(PHIGRADCOMP);
      phi_gradients_compute_gpu();
      TIMER_stop(PHIGRADCOMP);

#else

      phi_compute_phi_site();

      TIMER_start(PHIHALO);
      phi_halo();
      TIMER_stop(PHIHALO);

      if (phi_nop() == 5) COLL_set_Q();

      TIMER_start(PHIGRADCOMP);
      phi_gradients_compute();
      TIMER_stop(PHIGRADCOMP);

#endif

      TIMER_stop(TIMER_PHI_GRADIENTS);


      if (phi_is_finite_difference()) {

#ifdef _GPU_
	info("phi_is_finite_difference not yet supported in GPU mode\n");
	exit(1);
#endif

	TIMER_start(TIMER_FORCE_CALCULATION);

	if (colloid_ntotal() == 0) {
	  phi_force_calculation();
	}
	else {
	  phi_force_colloid();
	  /* Fixes the velocity gradient tensor for beris edwards */
	  colloids_fix_swd();
	}
	TIMER_stop(TIMER_FORCE_CALCULATION);

	TIMER_start(TIMER_ORDER_PARAMETER_UPDATE);
	phi_update_dynamics();
	TIMER_stop(TIMER_ORDER_PARAMETER_UPDATE);

      }
    }

#ifdef _GPU_

    TIMER_start(TIMER_COLLIDE);
    collide_gpu();
    TIMER_stop(TIMER_COLLIDE);

#else

    TIMER_start(TIMER_COLLIDE);
    collide();
    TIMER_stop(TIMER_COLLIDE);

#endif
 



    model_le_apply_boundary_conditions();


    TIMER_start(TIMER_HALO_LATTICE);
    #ifdef _GPU_
    halo_swap_gpu();
    #else
    distribution_halo();
    #endif
    TIMER_stop(TIMER_HALO_LATTICE);



    /* Colloid bounce-back applied between collision and
     * propagation steps. */



    if (subgrid_on()) {
      subgrid_update();
    }
    else {

      TIMER_start(TIMER_BBL);

#ifdef _GPU_
      get_phi_from_gpu();
#endif


      bounce_back_on_links();
      wall_bounce_back();

      TIMER_stop(TIMER_BBL);
    }




    /* There must be no halo updates between bounce back
     * and propagation, as the halo regions are active */

#ifdef _GPU_
    TIMER_start(TIMER_PROPAGATE);
    propagation_gpu();
    TIMER_stop(TIMER_PROPAGATE);
#else
    TIMER_start(TIMER_PROPAGATE);
    propagation();
    TIMER_stop(TIMER_PROPAGATE);
#endif







    TIMER_stop(TIMER_STEPS);

    /* Configuration dump */

    if (is_config_step()) {

#ifdef _GPU_
      get_f_from_gpu();
#endif

      sprintf(filename, "dist-%8.8d", step);
      io_write(filename, distribution_io_info());
      sprintf(filename, "%s%8.8d", "config.cds", step);
      colloid_io_write(filename);
      sprintf(filename,"phi-%8.8d",step);
      io_write(filename, io_info_phi);
    }

    /* Measurements */


    if (is_measurement_step()) {	  

#ifdef _GPU_
      get_velocity_from_gpu();
      get_f_from_gpu();
#endif

      sprintf(filename, "%s%8.8d", "config.cds", step);
      colloid_io_write(filename);

      if (phi_nop() == 5) {
	info("Writing scalar order parameter file at step %d!\n", step);
	sprintf(filename,"qs_dir-%8.8d",step);
	io_write(filename, io_info_scalar_q_);
      }
    }



    if (is_shear_measurement_step()) {

#ifdef _GPU_
      get_velocity_from_gpu();
      get_f_from_gpu();
#endif


      stats_rheology_stress_profile_accumulate();
    }


    if (is_shear_output_step()) {

#ifdef _GPU_
      get_velocity_from_gpu();
      get_f_from_gpu();
#endif


      sprintf(filename, "str-%8.8d.dat", step);
      stats_rheology_stress_section(filename);
      stats_rheology_stress_profile_zero();
    }

    if (is_phi_output_step()) {
      info("Writing phi file at step %d!\n", step);
      sprintf(filename,"phi-%8.8d",step);
      io_write(filename, io_info_phi);
    }


    if (is_vel_output_step()) {
#ifdef _GPU_
      get_velocity_from_gpu();
#endif
      info("Writing velocity output at step %d!\n", step);
      sprintf(filename, "vel-%8.8d", step);
      io_write(filename, io_info_velocity_);
    }

    /* Print progress report */

    if (is_statistics_step()) {

#ifdef _GPU_
      get_velocity_from_gpu();
      get_f_from_gpu();
#endif


      stats_distribution_print();
      phi_stats_print_stats();
      stats_free_energy_density();
      ludwig_report_momentum();
      hydrodynamics_stats();

      test_isothermal_fluctuations();
      info("\nCompleted cycle %d\n", step);
    }

    stats_calibration_accumulate(step);

    /* Next time step */
  }


#ifdef _GPU_
  get_velocity_from_gpu();
  get_f_from_gpu();
  finalise_gpu();
#endif


  /* Dump the final configuration if required. */

  if (is_config_at_end()) {
    sprintf(filename, "dist-%8.8d", step);
    io_write(filename, distribution_io_info());
    sprintf(filename, "%s%8.8d", "config.cds", step);
    colloid_io_write(filename);
    if (phi_nop()) {
      sprintf(filename,"phi-%8.8d",step);
      io_write(filename, io_info_phi);
    }
  }

  /* Shut down cleanly. Give the timer statistics. Finalise PE. */

  stats_rheology_finish();
  stats_turbulent_finish();
  stats_calibration_finish();
  colloids_finish();
  wall_finish();

  TIMER_stop(TIMER_TOTAL);
  TIMER_statistics();

  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  ludwig_report_momentum
 *
 *  Tidy report of the current momentum of the system.
 *
 *****************************************************************************/

void ludwig_report_momentum(void) {

  int n;

  double g[3];         /* Fluid momentum (total) */
  double gc[3];        /* Colloid momentum (total) */
  double gwall[3];     /* Wall momentum (for accounting purposes only) */
  double gtotal[3];

  for (n = 0; n < 3; n++) {
    gtotal[n] = 0.0;
    g[n] = 0.0;
    gc[n] = 0.0;
    gwall[n] = 0.0;
  }

  stats_distribution_momentum(g);
  stats_colloid_momentum(gc);
  if (wall_present()) wall_net_momentum(gwall);

  for (n = 0; n < 3; n++) {
    gtotal[n] = g[n] + gc[n] + gwall[n];
  }

  info("\n");
  info("Momentum - x y z\n");
  info("[total   ] %14.7e %14.7e %14.7e\n", gtotal[X], gtotal[Y], gtotal[Z]);
  info("[fluid   ] %14.7e %14.7e %14.7e\n", g[X], g[Y], g[Z]);
  if (colloid_ntotal()) {
    info("[colloids] %14.7e %14.7e %14.7e\n", gc[X], gc[Y], gc[Z]);
  }
  if (wall_present()) {
    info("[walls   ] %14.7e %14.7e %14.7e\n", gwall[X], gwall[Y], gwall[Z]);
  }

  return;
}
