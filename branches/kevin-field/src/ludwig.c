/*****************************************************************************
 *
 *  ludwig.c
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
#include "propagation_ode.h"

#include "site_map.h"
#include "physics.h"
#include "cio.h"

#include "io_harness.h"
#include "phi.h"
#include "phi_stats.h"
#include "phi_force.h"
#include "phi_force_colloid.h"
#include "phi_fluctuations.h"
#include "phi_gradients.h"
#include "phi_lb_coupler.h"
#include "phi_update.h"
#include "phi_update_rt.h"
#include "phi_cahn_hilliard.h"
#include "blue_phase.h"
#include "model_le.h"
#include "colloids_Q_tensor.h"

#include "advection_rt.h"
#include "distribution_rt.h"
#include "collision_rt.h"
#include "gradient_rt.h"
#include "site_map_rt.h"
#include "blue_phase_rt.h"
#include "polar_active_rt.h"

#include "psi.h"
#include "psi_rt.h"
#include "psi_sor.h"
#include "psi_stats.h"
#include "psi_force.h"
#include "psi_colloid.h"
#include "nernst_planck.h"

#include "hydro_rt.h"

#include "stats_colloid.h"
#include "stats_turbulent.h"
#include "stats_surfactant.h"
#include "stats_rheology.h"
#include "stats_free_energy.h"
#include "stats_distribution.h"
#include "stats_calibration.h"
#include "stats_velocity.h"
#include "stats_sigma.h"
#include "stats_symmetric.h"

#include "ludwig.h"
#include "field.h"
#include "field_grad.h"

typedef struct ludwig_s ludwig_t;
struct ludwig_s {
  hydro_t * hydro;          /* Hydrodynamic quantities */
  field_t * phi;            /* Scalar order parameter */
  field_t * p;              /* Vector order parameter */
  field_t * q;              /* Tensor order parameter */
  field_grad_t * phi_grad;  /* Gradients for phi */
  field_grad_t * p_grad;    /* Gradients for p */
  field_grad_t * q_grad;    /* Gradients for q */
  psi_t * psi;              /* Electrokinetics */
};

static int ludwig_rt(ludwig_t * ludwig);
static void ludwig_report_momentum(void);
int free_energy_init_rt(ludwig_t * ludwig);
int symmetric_rt_initial_conditions(field_t * phi);

/*****************************************************************************
 *
 *  ludwig_rt
 *
 *  Digest the run-time arguments for different parts of the code.
 *
 *****************************************************************************/

static int ludwig_rt(ludwig_t * ludwig) {

  int p;
  int form;
  int n, nstat;
  char filename[FILENAME_MAX];
  char subdirectory[FILENAME_MAX];
  char value[BUFSIZ];
  int io_grid_default[3] = {1, 1, 1};
  int io_grid[3];
  io_info_t * iohandler = NULL;

  assert(ludwig);
  
  TIMER_init();
  TIMER_start(TIMER_TOTAL);

#ifdef OLD_PHI
  free_energy_run_time();
  phi_update_run_time();
  advection_run_time();

  coords_run_time();
#else
  /* Initialise free-energy related objects, and the coordinate
   * system (the halo extent depends on choice of free energy). */

  free_energy_init_rt(ludwig);
#endif

  init_control();

  init_physics();

#ifdef OLD_PHI
  le_init();
#else
  /* moved to free_energy_init_rt() with coords_init() */
#endif

  if (is_propagation_ode()) propagation_ode_init();

  distribution_run_time();
  collision_run_time();
  site_map_run_time();

  ran_init();

  psi_init_rt(&ludwig->psi);
  hydro_rt(&ludwig->hydro);

  /* PHI I/O */

  RUN_get_int_parameter_vector("default_io_grid", io_grid_default);
  for (n = 0; n < 3; n++) {
    io_grid[n] = io_grid_default[n];
  }
  RUN_get_int_parameter_vector("phi_io_grid", io_grid);

  form = IO_FORMAT_DEFAULT;
  n = RUN_get_string_parameter("phi_format", value, BUFSIZ);
  if (n != 0 && strcmp(value, "ASCII") == 0) {
    form = IO_FORMAT_ASCII;
  }


#ifdef OLD_PHI
    if (phi_fluctuations_on()) {
      int seed;
      RUN_get_int_parameter("fd_phi_fluctuations_seed", &p);
      seed = 0;
      if (p > 0) {
	seed = p;
	info("Order parameter noise seed: %u\n", seed);
      }
      phi_fluctuations_init(seed);
    }
  phi_init();
  phi_init_io_info(io_grid, form, form);

  info("\n");
  info("Order parameter I/O\n");
  info("-------------------\n");

  info("Order parameter I/O format:   %s\n", value);
  info("I/O decomposition:            %d %d %d\n", io_grid[0], io_grid[1],
       io_grid[2]);
  MODEL_init();
#else

  /* All the same I/O grid  */

  if (ludwig->phi) field_init_io_info(ludwig->phi, io_grid, form, form);
  if (ludwig->p) field_init_io_info(ludwig->p, io_grid, form, form);
  if (ludwig->q) field_init_io_info(ludwig->q, io_grid, form, form);

  if (ludwig->phi || ludwig->p || ludwig->q) {
    info("\n");
    info("Order parameter I/O\n");
    info("-------------------\n");
    
    info("Order parameter I/O format:   %s\n", value);
    info("I/O decomposition:            %d %d %d\n", io_grid[0], io_grid[1],
	 io_grid[2]);
    advection_run_time();
  }

  /* Can we move this down - the routine is in this file */
  if (ludwig->phi) symmetric_rt_initial_conditions(ludwig->phi);

#endif

  wall_init();
  COLL_init();

#ifdef OLD_PHI
  gradient_run_time();
#endif
  /* NOW INITIAL CONDITIONS */

  pe_subdirectory(subdirectory);

  if (get_step() == 0) {
    n = 0;
    distribution_rt_initial_conditions();
    RUN_get_int_parameter("LE_init_profile", &n);

    if (n != 0) model_le_init_shear_profile();
  }
  else {
    /* Distributions */

    sprintf(filename, "%sdist-%8.8d", subdirectory, get_step());
    info("Re-starting simulation at step %d with data read from "
	 "config\nfile(s) %s\n", get_step(), filename);

    io_read(filename, distribution_io_info());

#ifdef OLD_PHI
    if (phi_is_finite_difference()) {
      phi_io_info(&iohandler);
      assert(iohandler);
      sprintf(filename,"%sphi-%8.8d", subdirectory, get_step());
      info("Reading phi state from %s\n", filename);
      io_read(filename, iohandler);
    }
#else
    assert(0);
    /* Restart t != 0 */
#endif
  }

#ifdef OLD_PHI
  phi_gradients_init();
#else
  /* gradient initialisation for field stuff */
  if (ludwig->phi) gradient_rt_init(ludwig->phi_grad);
  if (ludwig->p) gradient_rt_init(ludwig->p_grad);
  if (ludwig->q) gradient_rt_init(ludwig->q_grad);
#endif

  stats_rheology_init();
  stats_turbulent_init();
  collision_init();

  /* Calibration statistics for ah required? */

  nstat = 0;
  n = RUN_get_string_parameter("calibration", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) nstat = 1;
  stats_calibration_init(nstat);

  /* Calibration of surface tension required (symmetric only) */

  nstat = 0;
  n = RUN_get_string_parameter("calibration_sigma", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) nstat = 1;
#ifdef OLD_PHI
  stats_sigma_init(nstat);
#else
  stats_sigma_init(ludwig->phi, nstat);
  /* One call to this before start of time steps? */
  if (distribution_ndist() == 2) phi_lb_from_field(ludwig->phi); 
#endif

  collision_init();

#ifdef OLD_PHI
  if (get_step() == 0 && phi_nop() == 5) {
    phi_io_info(&iohandler);
    blue_phase_rt_initial_conditions();
    /* To be replaced by io_write_data() */
    /* info("Writing order parameter file at step %d!\n", step);
    sprintf(filename,"%sphi-%8.8d", subdirectory, step);
    io_write(filename, iohandler);*/
  }
#else
  /* Initial Q_ab field required, apparently More GENERAL? */

  if (get_step() == 0 && ludwig->p) {
    polar_active_rt_initial_conditions();
  }
  if (get_step() == 0 && ludwig->q) {
    blue_phase_rt_initial_conditions(ludwig->q);
  }
#endif

  /* Electroneutrality */

  if (get_step() == 0 && ludwig->psi) {
    psi_colloid_rho_set(ludwig->psi);
    psi_colloid_electroneutral(ludwig->psi);
  }
2
  return 0;
}

/*****************************************************************************
 *
 *  ludwig_run
 *
 *****************************************************************************/

void ludwig_run(const char * inputfile) {

  char    filename[FILENAME_MAX];
  char    subdirectory[FILENAME_MAX];
  int     step = 0;
  int     is_subgrid = 0;
  double  fzero[3] = {0.0, 0.0, 0.0};
  io_info_t * iohandler = NULL;

  ludwig_t * ludwig = NULL;

  ludwig = calloc(1, sizeof(ludwig_t));
  assert(ludwig);

  pe_init();
  RUN_read_input_file(inputfile);

  ludwig_rt(ludwig);

  /* Report initial statistics */

  pe_subdirectory(subdirectory);

  info("Initial conditions.\n");
  stats_distribution_print();
#ifdef OLD_PHI
  phi_stats_print_stats();
#else
  if (distribution_ndist() == 2) phi_lb_to_field(ludwig->phi);
  if (ludwig->phi) stats_field_info(ludwig->phi);
  if (ludwig->p)   stats_field_info(ludwig->p);
  if (ludwig->q)   stats_field_info(ludwig->q);
#endif
  if (ludwig->psi) psi_stats_info(ludwig->psi);
  ludwig_report_momentum();

  /* Main time stepping loop */

  info("\n");
  info("Starting time step loop.\n");
  subgrid_on(&is_subgrid);

  while (next_step()) {

    TIMER_start(TIMER_STEPS);
    step = get_step();
    if (ludwig->hydro) hydro_f_zero(ludwig->hydro, fzero);

#ifdef OLD_PHI
    COLL_update(ludwig->hydro);
#else
    COLL_update(ludwig->hydro, ludwig->phi, ludwig->p, ludwig->q);
#endif
    /* Electrokinetics */

    if (ludwig->psi) {
      psi_colloid_rho_set(ludwig->psi);
      psi_halo_psi(ludwig->psi);
      /* Sum force for this step before update */
      psi_force_grad_mu(ludwig->psi, ludwig->hydro);
      psi_sor_poisson(ludwig->psi);
      psi_halo_rho(ludwig->psi);
      /* u halo should not be repeated if phi active... */ 
      if (ludwig->hydro) hydro_u_halo(ludwig->hydro);
      nernst_planck_driver(ludwig->psi, ludwig->hydro);
    }

    /* Order parameter */

#ifdef OLD_PHI
    if (phi_nop()) {

      TIMER_start(TIMER_PHI_GRADIENTS);

      /* Note that the liquid crystal boundary conditions must come after
       * the halo swap, but before the gradient calculation. */

      phi_compute_phi_site();
      phi_halo();
      phi_gradients_compute();
      if (phi_nop() == 5) blue_phase_redshift_compute();
#else
      /* if symmetric_lb store phi to field */
      if (distribution_ndist() == 2) phi_lb_to_field(ludwig->phi);

      if (ludwig->phi) {
	field_halo(ludwig->phi);
	field_grad_compute(ludwig->phi_grad);
      }
      if (ludwig->q) {
	field_halo(ludwig->q);
	field_grad_compute(ludwig->q_grad);
	blue_phase_redshift_compute(); /* if redshift required? */
      }
#endif

      TIMER_stop(TIMER_PHI_GRADIENTS);

#ifdef OLD_PHI
      if (phi_is_finite_difference()) {

	TIMER_start(TIMER_FORCE_CALCULATION);

	if (colloid_ntotal() == 0) {
	  phi_force_calculation(ludwig->hydro);
	}
	else {
	  phi_force_colloid(ludwig->hydro);
	}
	TIMER_stop(TIMER_FORCE_CALCULATION);

	TIMER_start(TIMER_ORDER_PARAMETER_UPDATE);
	phi_update_dynamics(ludwig->hydro);
	TIMER_stop(TIMER_ORDER_PARAMETER_UPDATE);

      }
    }
#else
    /* order parameter dynamics (not if symmetric_lb) */

    if (distribution_ndist() == 2) {
      /* dynamics are dealt with at the collision stage (below) */
    }
    else {

      TIMER_start(TIMER_FORCE_CALCULATION);

      if (colloid_ntotal() == 0) {
	phi_force_calculation(ludwig->phi, ludwig->hydro);
      }
      else {
	phi_force_colloid(ludwig->hydro);
      }
      TIMER_stop(TIMER_FORCE_CALCULATION);

      if (ludwig->phi) phi_cahn_hilliard(ludwig->phi, ludwig->hydro);
      if (ludwig->p) assert(0);
      if (ludwig->q) blue_phase_beris_edwards(ludwig->q, ludwig->hydro);
    }
#endif

    /* Collision stage */

    TIMER_start(TIMER_COLLIDE);
    collide(ludwig->hydro);
    TIMER_stop(TIMER_COLLIDE);

    model_le_apply_boundary_conditions();

    TIMER_start(TIMER_HALO_LATTICE);
    distribution_halo();
    TIMER_stop(TIMER_HALO_LATTICE);

    /* Colloid bounce-back applied between collision and
     * propagation steps. */

    if (is_subgrid) {
      subgrid_update(ludwig->hydro);
    }
    else {
      TIMER_start(TIMER_BBL);
      wall_update();
      bounce_back_on_links();
      wall_bounce_back();
      TIMER_stop(TIMER_BBL);
    }

    /* There must be no halo updates between bounce back
     * and propagation, as the halo regions are active */

    TIMER_start(TIMER_PROPAGATE);

    if(is_propagation_ode()) {
      propagation_ode();
    }
    else {
      propagation();
    }

    TIMER_stop(TIMER_PROPAGATE);

    TIMER_stop(TIMER_STEPS);

    /* Configuration dump */

    if (is_config_step()) {
      info("Writing distribution output at step %d!\n", step);
      sprintf(filename, "%sdist-%8.8d", subdirectory, step);
      io_write(filename, distribution_io_info());
    }

    /* is_measurement_step() is here to prevent 'breaking' old input
     * files; it should really be removed. */

    if (is_config_step() || is_measurement_step() || is_colloid_io_step()) {
      if (colloid_ntotal() > 0) {
	info("Writing colloid output at step %d!\n", step);
	sprintf(filename, "%s%s%8.8d", subdirectory, "config.cds", step);
	colloid_io_write(filename);
      }
    }

    if (is_phi_output_step() || is_config_step()) {
#ifdef OLD_PHI
      if (phi_nop() > 0) {
	phi_io_info(&iohandler);
	info("Writing phi file at step %d!\n", step);
	sprintf(filename,"%sphi-%8.8d", subdirectory, step);
	io_write(filename, iohandler);
      }
#else
      if (ludwig->phi) {
	field_io_info(ludwig->phi, &iohandler);
	info("Writing phi file at step %d!\n", step);
	sprintf(filename,"%sphi-%8.8d", subdirectory, step);
	io_write_data(iohandler, filename, ludwig->phi);
      }
      if (ludwig->q) {
	field_io_info(ludwig->q, &iohandler);
	info("Writing qs file at step %d!\n", step);
	sprintf(filename,"%sqs-%8.8d", subdirectory, step);
	io_write_data(iohandler, filename, ludwig->q);
      }
#endif
    }

    if (is_psi_output_step()) {
      if (ludwig->psi) {
	psi_io_info(ludwig->psi, &iohandler);
	info("Writing psi file at step %d!\n", step);
	sprintf(filename,"%spsi-%8.8d", subdirectory, step);
	io_write_data(iohandler, filename, ludwig->psi);
      }
    }

    /* Measurements */

    if (is_measurement_step()) {	  
      stats_sigma_measure(ludwig->phi, step);
    }

    if (is_shear_measurement_step()) {
      stats_rheology_stress_profile_accumulate(ludwig->hydro);
    }

    if (is_shear_output_step()) {
      sprintf(filename, "%sstr-%8.8d.dat", subdirectory, step);
      stats_rheology_stress_section(filename);
      stats_rheology_stress_profile_zero();
    }

    if (is_vel_output_step()) {
      hydro_io_info(ludwig->hydro, &iohandler);
      info("Writing velocity output at step %d!\n", step);
      sprintf(filename, "%svel-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->hydro);
    }

    /* Print progress report */

    if (is_statistics_step()) {

      stats_distribution_print();
#ifdef OLD_PHI
      if (phi_nop()) {
	phi_stats_print_stats();
	stats_free_energy_density();
      }
#else
      if (distribution_ndist() == 2) phi_lb_to_field(ludwig->phi);
      if (ludwig->phi) stats_field_info(ludwig->phi);
      if (ludwig->p)   stats_field_info(ludwig->p);
      if (ludwig->q)   stats_field_info(ludwig->q);
      stats_free_energy_density(ludwig->q);
#endif
      if (ludwig->psi) {
	psi_stats_info(ludwig->psi);
      }
      ludwig_report_momentum();
      if (ludwig->hydro) stats_velocity_minmax(ludwig->hydro);

      test_isothermal_fluctuations();
      info("\nCompleted cycle %d\n", step);
    }

    stats_calibration_accumulate(step, ludwig->hydro);

    /* Next time step */
  }

  /* To prevent any conflict between the last regular dump, and
   * a final dump, there's a barrier here. */

  MPI_Barrier(pe_comm()); 

  /* Dump the final configuration if required. */

  if (is_config_at_end()) {
    sprintf(filename, "%sdist-%8.8d", subdirectory, step);
    io_write(filename, distribution_io_info());
    sprintf(filename, "%s%s%8.8d", subdirectory, "config.cds", step);
    colloid_io_write(filename);
#ifdef OLD_PHI
    if (phi_nop()) {
      phi_io_info(&iohandler);
      sprintf(filename,"%sphi-%8.8d", subdirectory, step);
    }
#else
    if (ludwig->phi) {
      field_io_info(ludwig->phi, &iohandler);
      info("Writing phi file at step %d!\n", step);
      sprintf(filename,"%sphi-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->phi);
    }
    if (ludwig->q) {
      field_io_info(ludwig->q, &iohandler);
      info("Writing qs file at step %d!\n", step);
      sprintf(filename,"%sqs-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->q);
    }
#endif
  }

  /* Shut down cleanly. Give the timer statistics. Finalise PE. */

  stats_rheology_finish();
  stats_turbulent_finish();
  stats_calibration_finish();
  colloids_finish();
  wall_finish();

  if (ludwig->phi_grad) field_grad_free(ludwig->phi_grad);
  if (ludwig->p_grad)   field_grad_free(ludwig->p_grad);
  if (ludwig->q_grad)   field_grad_free(ludwig->q_grad);
  if (ludwig->phi)      field_free(ludwig->phi);
  if (ludwig->p)        field_free(ludwig->p);
  if (ludwig->q)        field_free(ludwig->q);

  TIMER_stop(TIMER_TOTAL);
  TIMER_statistics();

  pe_finalise();

  return;
}

/*****************************************************************************
 *
 *  ludwig_report_momentum
 *
 *  Tidy report of the current momentum of the system.
 *
 *****************************************************************************/

static void ludwig_report_momentum(void) {

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

/******************************************************************************
 *
 *  free_energy_init_rt
 *
 *  This is to get over the rather awkward initialisation of the
 *  order parameter, free energy, and coordinate system, which
 *  are inter-related.
 *
 *  The choice of free energy sets the maximum halo width required,
 *  and what order parameter is required. This in turn determines
 *  the extent of the coordinate system, which must be initialised
 *  before the order parameter field, which is required to compute
 *  the free energy.
 *
 *  No free energy is appropriate for a single phase fluid.
 *
 *  This is currently rather repetative, so some rationalisation
 *  is required.
 *
 *  TODO: force divergence method alters halo requirement?
 *
 *****************************************************************************/

#ifdef OLD_PHI
#else

int free_energy_init_rt(ludwig_t * ludwig) {

  int n = 0;
  int order;
  int p;
  int nf;
  int ngrad;
  int nhalo;
  double value;
  char description[BUFSIZ];
  unsigned int seed;

  assert(ludwig);

  n = RUN_get_string_parameter("free_energy", description, BUFSIZ);

  if (strcmp(description, "none") == 0) {
    /* Appropriate for single fluid */
    info("\n");
    info("No free energy selected\n");
    phi_force_required_set(0); /* Could reverse the default */

    nhalo = 1;
    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
  }
  else if (strcmp(description, "symmetric") == 0 ||
	   strcmp(description, "symmetric_noise") == 0) {

    /* Symmetric free energy via finite difference */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 2;   /* Require stress divergence. */
    ngrad = 2;   /* \nabla^2 required */

    /* Noise requires additional stencil point for Cahn Hilliard */

    if (strcmp(description, "symmetric_noise") == 0) {
      nhalo = 3;
    }

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "phi", &ludwig->phi);
    field_init(ludwig->phi, nhalo);
    field_grad_create(ludwig->phi, ngrad, &ludwig->phi_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
    symmetric_run_time();
    symmetric_phi_set(ludwig->phi, ludwig->phi_grad);

    info("\n");
    info("Using Cahn-Hilliard finite difference solver.\n");

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());

    /* This could be set via symmetric_noise, no? Or check consistent */
    p = 0;
    RUN_get_int_parameter("fd_phi_fluctuations", &p);
    info("Order parameter noise = %3s\n", (p == 0) ? "off" : " on");
    if (p != 0) phi_fluctuations_on_set(p);

    if (phi_fluctuations_on()) {
      if (nhalo != 3) fatal("Fluctuations: use symmetric_noise\n");
      RUN_get_int_parameter("fd_phi_fluctuations_seed", &p);
      seed = 0;
      if (p > 0) {
	seed = p;
	info("Order parameter noise seed: %u\n", seed);
      }
      phi_fluctuations_init(seed);
    }

    /* Force */

    p = 1; /* Default is to use divergence method */
    RUN_get_int_parameter("fd_force_divergence", &p);
    info("Force calculation:      %s\n",
         (p == 0) ? "phi grad mu method" : "divergence method");
    phi_force_divergence_set(p);

  }
  else if (strcmp(description, "symmetric_lb") == 0) {

    /* Symmetric free energy via full lattice kintic equation */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 1;   /* Require one piont for LB. */
    ngrad = 2;   /* \nabla^2 required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "phi", &ludwig->phi);
    field_init(ludwig->phi, nhalo);
    field_grad_create(ludwig->phi, ngrad, &ludwig->phi_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
    symmetric_run_time();
    symmetric_phi_set(ludwig->phi, ludwig->phi_grad);

    info("\n");
    info("Using full lattice Boltzmann solver for Cahn-Hilliard:\n");
    phi_force_required_set(0);

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());

  }
  else if (strcmp(description, "brazovskii") == 0) {

    /* Brazovskii (always finite difference). */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 3;   /* Required for stress diveregnce. */
    ngrad = 4;   /* (\nabla^2)^2 required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "phi", &ludwig->phi);
    field_init(ludwig->phi, nhalo);
    field_grad_create(ludwig->phi, ngrad, &ludwig->phi_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
    brazovskii_run_time();
    brazovskii_phi_set(ludwig->phi, ludwig->phi_grad);

    info("\n");
    info("Using Cahn-Hilliard solver:\n");

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());

    p = 1;
    RUN_get_int_parameter("fd_force_divergence", &p);
    info("Force caluclation:      %s\n",
         (p == 0) ? "phi grad mu method" : "divergence method");
    phi_force_divergence_set(p);


  }
  else if (strcmp(description, "surfactant") == 0) {
    /* Disable surfactant for the time being */
    info("Surfactant free energy is disabled\n");
    assert(0);
  }
  else if (strcmp(description, "lc_blue_phase") == 0) {

    /* Brazovskii (always finite difference). */

    nf = NQAB;   /* Tensor order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "q", &ludwig->q);
    field_init(ludwig->q, nhalo);
    field_grad_create(ludwig->q, ngrad, &ludwig->q_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");

    blue_phase_run_time();
    blue_phase_q_set(ludwig->q, ludwig->q_grad);

    info("\n");
    info("Using Beris-Edwards solver:\n");

    p = RUN_get_double_parameter("lc_Gamma", &value);
    if (p != 0) {
      blue_phase_be_set_rotational_diffusion(value);
      info("Rotational diffusion constant = %12.5e\n", value);
    }

  }
  else if (strcmp(description, "polar_active") == 0) {

    /* Polar active. */

    nf = NVECTOR;/* Vector order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "p", &ludwig->p);
    field_init(ludwig->p, nhalo);
    field_grad_create(ludwig->p, ngrad, &ludwig->p_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");

    polar_active_run_time();
    polar_active_p_set(ludwig->p, ludwig->p_grad);

    assert(0);
  }
  else {
    if (n == 1) {
      /* The user has put something which hasn't been recognised,
       * suggesting a spelling mistake */
      info("free_energy %s not recognised.\n", description);
      fatal("Please check and try again.\n");
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  symmetric_rt_initial_conditions
 *
 *  To be moved to symmtric_rt.c
 *
 *****************************************************************************/

int symmetric_rt_initial_conditions(field_t * phi) {

  int p;
  int ic, jc, kc, index;
  int ntotal[3], nlocal[3];
  int offset[3];
  double phi0, phi1;
  double noise0 = 0.1; /* Default value. */
  char value[BUFSIZ];

  assert(phi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(offset);
  ntotal[X] = N_total(X);
  ntotal[Y] = N_total(Y);
  ntotal[Z] = N_total(Z);

  phi0 = get_phi0();
  RUN_get_double_parameter("noise", &noise0);

  /* Default initialisation (always?) Note serial nature of this,
   * which could be replaced. */

  for (ic = 1; ic <= ntotal[X]; ic++) {
    for (jc = 1; jc <= ntotal[Y]; jc++) {
      for (kc = 1; kc <= ntotal[Z]; kc++) {

	phi1 = phi0 + noise0*(ran_serial_uniform() - 0.5);

	/* For computation with single fluid and no noise */
	/* Only set values if within local box */
	if ( (ic > offset[X]) && (ic <= offset[X] + nlocal[X]) &&
	     (jc > offset[Y]) && (jc <= offset[Y] + nlocal[Y]) &&
	     (kc > offset[Z]) && (kc <= offset[Z] + nlocal[Z]) ) {

	    index = coords_index(ic-offset[X], jc-offset[Y], kc-offset[Z]);
	    field_scalar_set(phi, index, phi1);
	    
	}
      }
    }
  }

  p = RUN_get_string_parameter("phi_initialisation", value, BUFSIZ);

  if (p != 0 && strcmp(value, "block") == 0) {
    info("Initialisng phi as block\n");
    phi_init_block(symmetric_interfacial_width());
  }

  if (p != 0 && strcmp(value, "bath") == 0) {
    info("Initialising phi for bath\n");
    phi_init_bath();
  }

  if (p != 0 && strcmp(value, "drop") == 0) {
    info("Initialising droplet\n");
    phi_lb_init_drop(0.4*L(X), symmetric_interfacial_width());
  }

  if (p != 0 && strcmp(value, "from_file") == 0) {
    info("Initial order parameter requested from file\n");
    info("Reading phi from serial file\n");

    assert(0);
    /* need to do something! */
  }

  if (distribution_ndist() == 2) phi_lb_from_field(phi);

  return 0;
}

#endif
