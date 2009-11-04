/*****************************************************************************
 *
 *  Ludwig
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

#include "pe.h"
#include "runtime.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
#include "control.h"
#include "free_energy_rt.h"
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
#include "brownian.h"
#include "ccomms.h"

#include "lattice.h"
#include "cio.h"
#include "regsteer.h"

#include "io_harness.h"
#include "phi.h"
#include "phi_stats.h"
#include "blue_phase.h"
#include "model_le.h"

#include "stats_turbulent.h"
#include "stats_surfactant.h"
#include "stats_rheology.h"

void set_block(void);

int main( int argc, char **argv )
{
  char    filename[FILENAME_MAX];
  int     step = 0;
  int     n;

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

  TIMER_init();
  TIMER_start(TIMER_TOTAL);

  ran_init();
  RAND_init_fluctuations();
  le_init();

  MODEL_init();
  wall_init();
  COLL_init();

  free_energy_run_time();

  if (get_step() == 0) {
    n = 0;
    RUN_get_int_parameter("LE_init_profile", &n);

    if (n != 0) model_le_init_shear_profile();
  }
  else {
    if (phi_is_finite_difference()) {
      sprintf(filename,"phi-%6.6d", get_step());
      info("Reading phi state from %s\n", filename);
      io_read(filename, io_info_phi);
    }
  }

  stats_rheology_init();
  stats_turbulent_init();

  /* Report initial statistics */

  TEST_statistics();
  TEST_momentum();
  phi_stats_print_stats();


  /* Main time stepping loop */

  while (next_step()) {

    TIMER_start(TIMER_STEPS);
    step = get_step();

#ifdef _BROWNIAN_
    brownian_set_random();
    CCOM_halo_particles();
    COLL_forces();
    brownian_step_no_inertia();
    cell_update();
#else
    hydrodynamics_zero_force();
    COLL_update();
    wall_update();

    /* Collision stage */
    collide();
    model_le_apply_boundary_conditions();
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
#endif

    TIMER_stop(TIMER_STEPS);

    /* Configuration dump */

    if (is_config_step()) {
      get_output_config_filename(filename, step);
      io_write(filename, io_info_distribution_);
      sprintf(filename, "%s%6.6d", "config.cds", step);
      CIO_write_state(filename);
    }

    /* Measurements */

    if (is_measurement_step()) {	  
      sprintf(filename, "%s%6.6d", "config.cds", step);
      /*CIO_write_state(filename);*/
    }

    if (is_shear_measurement_step()) {
      stats_rheology_stress_profile_accumulate();
    }

    if (is_shear_output_step()) {
      sprintf(filename, "str-%8.8d.dat", step);
      stats_rheology_stress_section(filename);
      stats_rheology_stress_profile_zero();
      stats_rheology_mean_stress("stress_means.dat");
    }

    if (is_phi_output_step()) {
      info("Writing phi file at step %d!\n", step);
      sprintf(filename,"phi-%6.6d",step);
      io_write(filename, io_info_phi);
    }

    if (is_vel_output_step()) {
      info("Writing velocity output at step %d!\n", step);
      sprintf(filename, "vel-%6.6d", step);
      io_write(filename, io_info_velocity_);
    }

    /* Print progress report */

    if (is_statistics_step()) {

#ifndef _BROWNIAN_
      /* PENDING TODO MISC_curvature(); */
      TEST_statistics();
      TEST_momentum();
      wall_force();
      hydrodynamics_stats();
#endif
#ifdef _NOISE_
      TEST_fluid_temperature();
#endif
      phi_stats_print_stats();
      info("\nCompleted cycle %d\n", step);
    }

    /* Next time step */
  }

  /* Dump the final configuration if required. */

  if (is_config_at_end()) {
    get_output_config_filename(filename, step);
    io_write(filename, io_info_distribution_);
    sprintf(filename, "%s%6.6d", "config.cds", step);
    CIO_write_state(filename);
    sprintf(filename,"phi-%6.6d",step);
    io_write(filename, io_info_phi);
  }

  /* Shut down cleanly. Give the timer statistics. Finalise PE. */

  stats_rheology_finish();
  stats_turbulent_finish();
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

    index = get_site_index(ic, jc, kc);
    rho = get_rho_at_site(index);
    get_momentum_at_site(index, u);

    printf("%4d %10.8f %10.8f\n", ic, rho, u[Y]/rho);
  }

  return;
}

void set_block() {

  int index;
  int ic, jc, kc;
  int N[ND];
  double phi;

  get_N_local(N);

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	phi = -1.0;
	if (ic >=1 && ic < 16) phi =1.0;

	index = get_site_index(ic, jc, kc);
	set_phi(phi, index);
      }
    }
  }

  return;
}
