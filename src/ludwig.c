/*****************************************************************************
 *
 *  ludwig.c
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Coordinate system, general */
#include "pe.h"
#include "runtime.h"
#include "ran.h"
#include "noise.h"
#include "timer.h"
#include "coords_rt.h"
#include "coords.h"
#include "leesedwards_rt.h"
#include "control.h"
#include "util.h"
#include "util_bits.h"

#include "model_le.h"
#include "bbl.h"

#include "collision.h"
#include "propagation.h"
#include "distribution_rt.h"
#include "collision_rt.h"

#include "map_rt.h"
#include "wall_rt.h"
#include "interaction.h"
#include "physics_rt.h"

#include "hydro_rt.h"

#include "io_info_args_rt.h"

#include "phi_stats.h"
#include "phi_force.h"
#include "phi_force_colloid.h"
#include "phi_lb_coupler.h"

/* Order parameter fields */
#include "field.h"
#include "field_grad.h"
#include "gradient_rt.h"

/* Free energy */
#include "symmetric_rt.h"
#include "brazovskii_rt.h"
#include "surfactant_rt.h"
#include "polar_active_rt.h"
#include "blue_phase_rt.h"
#include "lc_droplet_rt.h"
#include "fe_ternary_rt.h"
#include "fe_electro.h"
#include "fe_electro_symmetric.h"

#include "fe_force_method_rt.h"

/* Dynamics */
#include "cahn_hilliard.h"
#include "phi_cahn_hilliard.h"
#include "cahn_hilliard_stats.h"
#include "leslie_ericksen.h"
#include "blue_phase_beris_edwards.h"

/* Colloids */
#include "colloids_rt.h"
#include "colloid_sums.h"
#include "colloids_halo.h"
#include "build.h"
#include "subgrid.h"
#include "colloids.h"
#include "advection_rt.h"

/* Viscosity model */
#include "visc.h"
#include "visc_arrhenius.h"

/* Open boundary conditions */
#include "lb_bc_open_rt.h"
#include "phi_bc_open_rt.h"

/* Electrokinetics */
#include "psi.h"
#include "psi_rt.h"
#include "psi_stats.h"
#include "psi_force.h"
#include "psi_solver.h"
#include "psi_colloid.h"
#include "nernst_planck.h"

/* Statistics */
#include "stats_colloid.h"
#include "stats_colloid_force_split.h"
#include "stats_turbulent.h"
#include "stats_surfactant.h"
#include "stats_rheology.h"
#include "stats_free_energy.h"
#include "stats_distribution.h"
#include "stats_calibration.h"
#include "stats_velocity.h"
#include "stats_sigma.h"
#include "stats_symmetric.h"

#include "fe_lc_stats.h"
#include "fe_ternary_stats.h"

#include "ludwig.h"

typedef struct ludwig_s ludwig_t;
struct ludwig_s {
  pe_t * pe;                /* Parallel environment */
  rt_t * rt;                /* Run time input handler */
  cs_t * cs;                /* Coordinate system */
  physics_t * phys;         /* Physical parameters */
  lees_edw_t * le;          /* Lees Edwards sliding periodic boundaries */
  lb_t * lb;                /* Lattice Botlzmann */
  hydro_t * hydro;          /* Hydrodynamic quantities */
  field_t * phi;            /* Scalar order parameter */
  field_t * p;              /* Vector order parameter */
  field_t * q;              /* Tensor order parameter */
  field_grad_t * phi_grad;  /* Gradients for phi */
  field_grad_t * p_grad;    /* Gradients for p */
  field_grad_t * q_grad;    /* Gradients for q */
  psi_t * psi;              /* Electrokinetics */
  map_t * map;              /* Site map for fluid/solid status etc. */
  wall_t * wall;            /* Side walls / Porous media */
  noise_t * noise;          /* Generator for fluctuations */

  psi_solver_t * poisson;      /* Poisson solver */

  fe_t * fe;                   /* Free energy "polymorphic" version */
  ch_t * ch;                   /* Cahn Hilliard (surfactants) */
  phi_ch_t * pch;              /* Cahn Hilliard dynamics (binary fluid) */
  leslie_ericksen_t * leslie;  /* Leslie Ericksen Dynamics */
  beris_edw_t * be;            /* Beris Edwards dynamics */
  pth_t * pth;                 /* Thermodynamic stress/force calculation */
  fe_lc_t * fe_lc;             /* LC free energy */
  fe_symm_t * fe_symm;         /* Symmetric free energy */
  fe_surf_t * fe_surf;         /* Surfactant (van der Graf etc) */
  fe_ternary_t * fe_ternary;   /* Ternary (Semprebon et al.) */
  fe_brazovskii_t * fe_braz;   /* Brazovskki */

  visc_t * visc;               /* Viscosity model */

  colloids_info_t * collinfo;  /* Colloid information */
  colloid_io_t * cio;          /* Colloid I/O harness */
  ewald_t * ewald;             /* Ewald sum for dipoles */
  interact_t * interact;       /* Colloid-colloid interaction handler */
  bbl_t * bbl;                 /* Bounce-back on links boundary condition */

  lb_bc_open_t * inflow;       /* Inflow open boundary conidition (fluid) */
  lb_bc_open_t * outflow;      /* Outflow boundary condition (fluid) */
  phi_bc_open_t * phi_inflow;  /* Inflow (composition phi) */
  phi_bc_open_t * phi_outflow; /* Outflow (composition phi) */

  stats_sigma_t * stat_sigma;  /* Interfacial tension calibration */
  stats_ahydro_t * stat_ah;    /* Hydrodynamic radius calibration */
  stats_rheo_t * stat_rheo;    /* Rheology diagnostics */
  stats_turb_t * stat_turb;    /* Turbulent diagnostics */
  timekeeper_t tk;             /* Time keeper */
};

static int ludwig_rt(ludwig_t * ludwig);
static int ludwig_report_momentum(ludwig_t * ludwig);
static int ludwig_report_statistics(ludwig_t * ludwig, int itimestep);
static int ludwig_colloids_update(ludwig_t * ludwig);
static int ludwig_colloids_update_low_freq(ludwig_t * ludwig);

int ludwig_timekeeper_init(ludwig_t * ludwig);
int free_energy_init_rt(ludwig_t * ludwig);
int visc_model_init_rt(pe_t * pe, rt_t * rt, ludwig_t * ludwig);
int io_replace_values(field_t * field, map_t * map, int map_id, double value);
int io_replace_field_values(field_t * field, map_t * map, int status,
			    double value);

/*****************************************************************************
 *
 *  ludwig_rt
 *
 *  Digest the run-time arguments for different parts of the code.
 *
 *****************************************************************************/

static int ludwig_rt(ludwig_t * ludwig) {

  int ntstep;
  int n, nstat;
  char filename[FILENAME_MAX];

  pe_t * pe = NULL;
  cs_t * cs = NULL;
  rt_t * rt = NULL;

  assert(ludwig);

  TIMER_init(ludwig->pe);
  TIMER_start(TIMER_TOTAL);

  /* Prefer maximum L1 cache available on device */
  tdpAssert( tdpDeviceSetCacheConfig(tdpFuncCachePreferL1) );

  /* Initialise free-energy related objects, and the coordinate
   * system (the halo extent depends on choice of free energy). */

  physics_create(ludwig->pe, &ludwig->phys);
  free_energy_init_rt(ludwig);

  /* Just convenience shorthand */
  pe = ludwig->pe;
  cs = ludwig->cs;
  rt = ludwig->rt;

  ludwig_timekeeper_init(ludwig);
  init_control(pe, rt);

  physics_init_rt(rt, ludwig->phys);
  physics_info(pe, ludwig->phys);

  lb_run_time(pe, cs, rt, &ludwig->lb);
  collision_run_time(pe, rt, ludwig->lb);
  map_init_rt(pe, cs, rt, &ludwig->map);

  ran_init_rt(pe, rt);
  hydro_rt(pe, rt, cs, ludwig->le, &ludwig->hydro);
  visc_model_init_rt(pe, rt, ludwig);

  lb_bc_open_rt(pe, rt, cs, ludwig->lb, &ludwig->inflow, &ludwig->outflow);
  phi_bc_open_rt(pe, rt, cs, &ludwig->phi_inflow, &ludwig->phi_outflow);

  if (ludwig->phi || ludwig->p || ludwig->q) {
    advection_init_rt(pe, rt);
  }

  /* Can we move this down to t = 0 initialisation? */

  if (ludwig->fe_symm) {
    fe_symmetric_phi_init_rt(pe, rt, ludwig->fe_symm, ludwig->phi);
  }
  if (ludwig->fe_braz) {
    fe_brazovskii_phi_init_rt(pe, rt, ludwig->fe_braz, ludwig->phi);
  }
  if (ludwig->fe_surf) {
    fe_surf_phi_init_rt(pe, rt, ludwig->fe_surf, ludwig->phi);
    fe_surf_psi_init_rt(pe, rt, ludwig->fe_surf, ludwig->phi);
  }
  if (ludwig->fe_ternary) {
    fe_ternary_init_rt(pe, rt, ludwig->fe_ternary, ludwig->phi);
  }

  /* To be called before wall_rt_init() */
  if (ludwig->psi) {
    advection_init_rt(pe, rt);
    psi_rt_init_rho(pe, rt, ludwig->psi, ludwig->map);
  }

  wall_rt_init(pe, cs, rt, ludwig->lb, ludwig->map, &ludwig->wall);
  colloids_init_rt(pe, rt, cs, &ludwig->collinfo, &ludwig->cio,
		   &ludwig->interact, ludwig->wall, ludwig->map,
		   &ludwig->lb->model);
  colloids_init_ewald_rt(pe, rt, cs, ludwig->collinfo, &ludwig->ewald);

  bbl_create(pe, ludwig->cs, ludwig->lb, &ludwig->bbl);
  bbl_active_set(ludwig->bbl, ludwig->collinfo);
  {
    /* Kludge: this switch is unlikely to be required as the default
     * method of quaternions (ellipsoid_didt = 0) is exact */
    int ellipsoid_didt = rt_switch(rt, "bbl_ellipsoid_update_fd");
    bbl_didt_method_set(ludwig->bbl, ellipsoid_didt);
  }

  /* If any noise required ... */
  if (ludwig->lb->param->noise || (ludwig->pch && ludwig->pch->info.noise)) {
    noise_options_t opts = noise_options_default();
    noise_create(pe, cs, &opts, &ludwig->noise);
  }

  /* NOW INITIAL CONDITIONS */

  ntstep = physics_control_timestep(ludwig->phys);

  if (ntstep == 0) {
    double rho0 = 1.0;
    lb_rt_initial_conditions(pe, rt, ludwig->lb, ludwig->phys);
    physics_rho0(ludwig->phys, &rho0);
    if (ludwig->hydro) hydro_rho0(ludwig->hydro, rho0);

    /* This should be relocated with LE plane input */
    if (rt_switch(ludwig->rt, "LE_init_profile")) {
      if (lees_edw_nplane_total(ludwig->le) == 0) {
	pe_info(ludwig->pe, "Cannot use LE_init_profile with no planes\n");
	pe_fatal(ludwig->pe, "Please check the input and try again\n");
      }
      lb_le_init_shear_profile(ludwig->lb, ludwig->le);
    }
  }
  else {
    /* Distributions */

    pe_info(pe, "Re-starting simulation at step %d with data read from file\n",
	    ntstep);
    {
      io_event_t event = {0};
      pe_info(pe, "Reading distribution files for step %d\n", ntstep);
      lb_io_read(ludwig->lb, ntstep, &event);
    }

    /* Restart t != 0 for order parameter */

    if (ludwig->phi) {
      io_event_t event = {0};
      pe_info(pe, "Reading phi files for step %d\n", ntstep);
      field_io_read(ludwig->phi, ntstep, &event);
    }

    if (ludwig->p) {
      io_event_t event = {0};
      pe_info(pe, "Reading p files for step %d\n", ntstep);
      field_io_read(ludwig->p, ntstep, &event);
    }

    if (ludwig->q) {
      io_event_t event = {0};
      pe_info(pe, "Reading q_ab files for step %d\n", ntstep);
      field_io_read(ludwig->q, ntstep, &event);
    }

    if (ludwig->hydro) {
      io_event_t event = {0};
      pe_info(pe, "Reading rho/vel files for step %d\n", ntstep);
      hydro_io_read(ludwig->hydro, ntstep, &event);
    }

    if (ludwig->psi) {
      io_event_t event1 = {0};
      io_event_t event2 = {0};
      pe_info(pe, "Reading electrokinetics files for step %d\n", ntstep);
      field_io_read(ludwig->psi->psi, ntstep, &event1);
      field_io_read(ludwig->psi->rho, ntstep, &event2);
    }
  }

  /* gradient initialisation for field stuff */

  if (ludwig->phi) {
    gradient_rt_init(pe, rt, "phi", ludwig->phi_grad, ludwig->map,
		     ludwig->collinfo);
  }
  if (ludwig->p) {
    gradient_rt_init(pe, rt, "p", ludwig->p_grad, ludwig->map,
		     ludwig->collinfo);
  }
  if (ludwig->q) {
    gradient_rt_init(pe, rt, "q", ludwig->q_grad, ludwig->map,
		     ludwig->collinfo);
  }

  stats_rheology_create(pe, cs, &ludwig->stat_rheo);
  stats_turbulent_create(pe, cs, &ludwig->stat_turb);

  /* Calibration statistics for ah required? */

  n = rt_string_parameter(rt, "calibration", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) {
    stats_ahydro_create(pe, cs, ludwig->collinfo, ludwig->hydro,
			ludwig->map, &ludwig->stat_ah);
  }

  /* Calibration of surface tension required (symmetric only) */

  nstat = 0;
  n = rt_string_parameter(rt, "calibration_sigma", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) nstat = 1;

  if (ntstep == 0) {
    if (nstat) stats_sigma_create(pe, cs, ludwig->fe_symm, ludwig->phi,
				  &ludwig->stat_sigma);
    lb_ndist(ludwig->lb, &n);
    if (n == 2) phi_lb_from_field(ludwig->phi, ludwig->lb);
  }

  /* Initial Q_ab field required */

  if (ntstep == 0 && ludwig->p) {
    polar_active_rt_initial_conditions(pe, rt, ludwig->p);
  }

  if (ntstep == 0 && ludwig->q) {
    blue_phase_rt_initial_conditions(pe, rt, cs, ludwig->fe_lc, ludwig->q);
  }

  if (ntstep == 0 && ludwig->psi) {
    psi_colloid_rho_set(ludwig->psi, ludwig->collinfo);
    pe_info(pe, "\nArranging initial charge neutrality.\n\n");
    psi_electroneutral(ludwig->psi, ludwig->map);
  }

  if (ludwig->pch && ludwig->phi) {
    if (ludwig->pch->info.conserve == 2) {
      /* 2 is correction method requiring a reference sum. */
      cahn_hilliard_stats_time0(ludwig->pch, ludwig->phi, ludwig->map);
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  ludwig_run
 *
 *****************************************************************************/

void ludwig_run(const char * inputfile) {

  char    filename[FILENAME_MAX];
  int     is_porous_media = 0;
  int     step = 0;
  int     is_pm = 0;
  int     ncolloid = 0;
  double  fzero[3] = {0.0, 0.0, 0.0};
  double  uzero[3] = {0.0, 0.0, 0.0};
  int     im, multisteps;
  int	  flag;

  ludwig_t * ludwig = NULL;
  MPI_Comm comm;

  stats_vel_t statvel = stats_vel_default();


  ludwig = (ludwig_t*) calloc(1, sizeof(ludwig_t));
  assert(ludwig);

  pe_create(MPI_COMM_WORLD, PE_VERBOSE, &ludwig->pe);
  pe_mpi_comm(ludwig->pe, &comm);

  {
    /* We currently assume one GPU per MPI task, and that GPU id
     * can be equated to rank in the shared (node) communicator. */
    MPI_Comm node_comm = MPI_COMM_NULL;
    int rank = -1;
    int node_rank = -1;
    int node_size = -1;
    int ndevice   = -1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    tdpAssert( tdpGetDeviceCount(&ndevice) );

    if (ndevice > 0 && ndevice < node_size) {
      pe_info(ludwig->pe,  "MPI tasks per node: %d\n", node_size);
      pe_info(ludwig->pe,  "GPUs per node:      %d\n", ndevice);
      pe_fatal(ludwig->pe, "Expecting at least one GPU per MPI task\n");
    }

    tdpAssert(tdpSetDevice(node_rank));
    MPI_Comm_free(&node_comm);
  }

  rt_create(ludwig->pe, &ludwig->rt);
  rt_read_input_file(ludwig->rt, inputfile);
  rt_info(ludwig->rt);

  ludwig_rt(ludwig);

  statvel.print_vol_flux = rt_switch(ludwig->rt, "stats_vel_print_vol_flux");

  /* Report initial statistics */

  /* Move initilaised data to target for initial conditions/time stepping */

  map_memcpy(ludwig->map, tdpMemcpyHostToDevice);
  lb_memcpy(ludwig->lb, tdpMemcpyHostToDevice);

  if (ludwig->phi) field_memcpy(ludwig->phi, tdpMemcpyHostToDevice);
  if (ludwig->p)   field_memcpy(ludwig->p, tdpMemcpyHostToDevice);
  if (ludwig->q)   field_memcpy(ludwig->q, tdpMemcpyHostToDevice);

  colloids_info_ntotal(ludwig->collinfo, &ncolloid);
  if (ncolloid) colloids_memcpy(ludwig->collinfo, tdpMemcpyHostToDevice);

  /* Lap timer: include initial statistics in first trip */
  TIMER_start(TIMER_LAP);

  pe_info(ludwig->pe, "Initial conditions.\n");
  wall_is_pm(ludwig->wall, &is_porous_media);

  ludwig_report_statistics(ludwig, 0);
  ludwig_report_momentum(ludwig);

  /* Main time stepping loop */

  pe_info(ludwig->pe, "\n");
  pe_info(ludwig->pe, "Starting time step loop.\n");

  /* sync tasks before main loop for timing purposes */
  MPI_Barrier(comm);

  while (physics_control_next_step(ludwig->phys)) {

    TIMER_start(TIMER_STEPS);

    step = physics_control_timestep(ludwig->phys);

    if (ludwig->hydro) {
      hydro_f_zero(ludwig->hydro, fzero);
    }

    if ((step % ludwig->collinfo->rebuild_freq) == 0) {
      ludwig_colloids_update(ludwig);
    }
    else {
      ludwig_colloids_update_low_freq(ludwig);
    }

    /* Order parameter gradients */

    TIMER_start(TIMER_PHI_GRADIENTS);

    /* if symmetric_lb store phi to field */


    lb_ndist(ludwig->lb, &im);

    if (im == 2) phi_lb_to_field(ludwig->phi, ludwig->lb);

    if (ludwig->phi) {

      TIMER_start(TIMER_PHI_HALO);
      field_halo(ludwig->phi);
      TIMER_stop(TIMER_PHI_HALO);

      /* Boundary conditions on phi after halo and
       * before gradient calculation. */

      if (ludwig->phi_inflow) {
	phi_bc_open_t * inflow = ludwig->phi_inflow;
	inflow->func->update(inflow, ludwig->phi);
      }
      if (ludwig->phi_outflow) {
	phi_bc_open_t * outflow = ludwig->phi_outflow;
	outflow->func->update(outflow, ludwig->phi);
      }

      field_grad_compute(ludwig->phi_grad);
    }

    if (ludwig->p) {
      field_halo(ludwig->p);
      field_grad_compute(ludwig->p_grad);
    }

    if (ludwig->q) {
      TIMER_start(TIMER_PHI_HALO);
      field_halo(ludwig->q);
      TIMER_stop(TIMER_PHI_HALO);

      field_grad_compute(ludwig->q_grad);
      fe_lc_redshift_compute(ludwig->cs, ludwig->fe_lc);
    }
    TIMER_stop(TIMER_PHI_GRADIENTS);
    if (ludwig->fe_lc) fe_lc_active_stress(ludwig->fe_lc);

    /* Update any open boundary flows (before any advection) */

    if (ludwig->inflow) {
      ludwig->inflow->func->update(ludwig->inflow, ludwig->hydro);
    }
    if (ludwig->outflow) {
      ludwig->outflow->func->update(ludwig->outflow, ludwig->hydro);
    }

    /* Electrokinetics (including electro/symmetric requiring above
     * gradients for phi) */

    if (ludwig->psi) {
      /* Set charge distribution according to updated map */
      psi_colloid_rho_set(ludwig->psi, ludwig->collinfo);

      /* Poisson solve */

      TIMER_start(TIMER_ELECTRO_POISSON);

      ludwig->poisson->impl->solve(ludwig->poisson, step);

      TIMER_stop(TIMER_ELECTRO_POISSON);

      if (ludwig->hydro) {
	TIMER_start(TIMER_HALO_LATTICE);
	hydro_u_halo(ludwig->hydro);
	TIMER_stop(TIMER_HALO_LATTICE);

	/* Work-around for gpu regression tests ... */
	hydro_memcpy(ludwig->hydro, tdpMemcpyDeviceToHost);
      }


      /* Time splitting for high electrokinetic diffusions in Nernst Planck */

      psi_multisteps(ludwig->psi, &multisteps);

      for (im = 0; im < multisteps; im++) {

	TIMER_start(TIMER_HALO_LATTICE);
	psi_halo_psi(ludwig->psi);
	psi_halo_psijump(ludwig->psi);
	psi_halo_rho(ludwig->psi);
	TIMER_stop(TIMER_HALO_LATTICE);

	/* Force calculation is only once per LB timestep */
	if (im == 0) {

	  TIMER_start(TIMER_FORCE_CALCULATION);
	  psi_force_method(ludwig->psi, &flag);

          /* Force input as gradient of chemical potential
                 with integrated momentum correction       */
	  if (flag == PSI_FORCE_GRADMU) {
	    psi_force_gradmu(ludwig->psi, ludwig->fe, ludwig->phi,
			     ludwig->hydro,
			     ludwig->map, ludwig->collinfo);
	  }

          /* Force calculation as divergence of stress tensor */
	  if (flag == PSI_FORCE_DIVERGENCE) {
	    psi_force_divstress(ludwig->psi, ludwig->fe, ludwig->hydro,
				ludwig->collinfo);
	  }
	  TIMER_stop(TIMER_FORCE_CALCULATION);

	}

	TIMER_start(TIMER_ELECTRO_NPEQ);
	nernst_planck_driver_d3qx(ludwig->psi, ludwig->fe, ludwig->hydro,
				  ludwig->map, ludwig->collinfo);
	TIMER_stop(TIMER_ELECTRO_NPEQ);

      }

      TIMER_start(TIMER_HALO_LATTICE);
      psi_halo_psi(ludwig->psi);
      psi_halo_psijump(ludwig->psi);
      psi_halo_rho(ludwig->psi);
      TIMER_stop(TIMER_HALO_LATTICE);

      if (ludwig->hydro) {
	/* Workaround for gpu regression tests ... */
	hydro_memcpy(ludwig->hydro, tdpMemcpyHostToDevice);
      }

      nernst_planck_adjust_multistep(ludwig->psi);
      psi_zero_mean(ludwig->psi);
    }

    /* order parameter dynamics (not if symmetric_lb) */

    lb_ndist(ludwig->lb, &im);
    if (im == 2) {
      /* dynamics are dealt with at the collision stage (below) */
    }
    else {

      TIMER_start(TIMER_FORCE_CALCULATION);

      if (ludwig->psi) {
	/* Force in electrokinetic models is computed above */
      }
      else {
	if (ncolloid == 0) {

	  /* LC-droplet requires partial body force input and momentum
           * correction. This correction, via hydro_correct_momentum(),
           * should not include the contributions from the divergence
           * of the stress, so is done before phi_force_calculation(). */

	  if (ludwig->fe && ludwig->fe->id == FE_LC_DROPLET) {

	    fe_lc_droplet_t * fe = (fe_lc_droplet_t *) ludwig->fe;

	    if (wall_present(ludwig->wall)) {
	      fe_lc_droplet_bodyforce_wall(fe, ludwig->le, ludwig->hydro,
		                           ludwig->map, ludwig->wall);
	    }
	    else {
	      fe_lc_droplet_bodyforce(fe, ludwig->hydro);
	    }

	    hydro_correct_momentum(ludwig->hydro);
	  }

	  /* Force calculation as divergence of stress tensor */

          phi_force_calculation(ludwig->pe, ludwig->cs, ludwig->le,
				ludwig->wall,
                                ludwig->pth, ludwig->fe, ludwig->map,
                                ludwig->phi, ludwig->hydro);

	  /* Ternary free energy gradmu requires of momentum correction
	     after force calculation */

	  if (ludwig->fe && ludwig->hydro && ludwig->fe->id == FE_TERNARY) {
            hydro_correct_momentum(ludwig->hydro);
	  }

	}
	else {
	  if (ludwig->pth->method == FE_FORCE_METHOD_STRESS_DIVERGENCE) {
	  pth_force_colloid(ludwig->pth, ludwig->fe, ludwig->collinfo,
			    ludwig->hydro, ludwig->map, ludwig->wall,
			    &ludwig->lb->model);
	  }
	  else {
	    /* Allow case with colloids using PHI_GRADMU_CORRECTION */
	    phi_force_calculation(ludwig->pe, ludwig->cs, ludwig->le,
				  ludwig->wall, ludwig->pth, ludwig->fe,
				  ludwig->map, ludwig->phi, ludwig->hydro);
	  }
	}
      }

      TIMER_stop(TIMER_FORCE_CALCULATION);

      if (ludwig->q && is_statistics_step()) {
	stats_colloid_force_split_update(ludwig->collinfo, ludwig->fe);
      }

      TIMER_start(TIMER_ORDER_PARAMETER_UPDATE);

      if (ludwig->ch) {
	ch_solver(ludwig->ch, ludwig->fe, ludwig->phi, ludwig->hydro,
		  ludwig->map);
      }

      if (ludwig->pch) {
	phi_cahn_hilliard(ludwig->pch, ludwig->fe, ludwig->phi,
			  ludwig->hydro,
			  ludwig->map, ludwig->noise);
      }

      if (ludwig->p) {
	leslie_ericksen_update(ludwig->leslie, ludwig->hydro);
      }

      if (ludwig->q) {
	if (ludwig->hydro) {
	  TIMER_start(TIMER_U_HALO);
 	  hydro_u_halo(ludwig->hydro);
	  TIMER_stop(TIMER_U_HALO);
	}

	beris_edw_update(ludwig->be, ludwig->fe, ludwig->q, ludwig->q_grad,
			 ludwig->hydro,
			 ludwig->collinfo, ludwig->map, ludwig->noise);
      }

      TIMER_stop(TIMER_ORDER_PARAMETER_UPDATE);
    }

    if (ludwig->hydro) {

      /* Zero velocity field here, as velocity at collision is used
       * at next time step for FD above. Strictly, we only need to
       * do this if velocity output is required in presence of
       * colloids to present non-zero u inside particles. */

      hydro_u_zero(ludwig->hydro, uzero);

      /* Viscosity computation */
      if (ludwig->visc) {
	ludwig->visc->func->update(ludwig->visc, ludwig->hydro);
      }

      /* Collision stage */

      TIMER_start(TIMER_COLLIDE);

      lb_collide(ludwig->lb, ludwig->hydro, ludwig->map, ludwig->noise,
		 ludwig->fe, ludwig->visc);

      TIMER_stop(TIMER_COLLIDE);


      /* Boundary conditions */

      if (ludwig->le) {
	lb_le_apply_boundary_conditions(ludwig->lb, ludwig->le);
      }

      TIMER_start(TIMER_HALO_LATTICE);

      lb_halo(ludwig->lb);

      TIMER_stop(TIMER_HALO_LATTICE);

      /* Open boundaries */

      if (ludwig->inflow) {
	lb_bc_open_t * inflow = ludwig->inflow;
	inflow->func->update(inflow, ludwig->hydro);
	inflow->func->impose(inflow, ludwig->hydro, ludwig->lb);
      }
      if (ludwig->outflow) {
	lb_bc_open_t * outflow = ludwig->outflow;
	outflow->func->update(outflow, ludwig->hydro);
	outflow->func->impose(outflow, ludwig->hydro, ludwig->lb);
      }

      /* Colloid bounce-back applied between collision and
       * propagation steps. */

      TIMER_start(TIMER_BBL);
      wall_set_wall_distributions(ludwig->wall);

      subgrid_update(ludwig->collinfo, ludwig->hydro, ludwig->lb->param->noise);
      bounce_back_on_links(ludwig->bbl, ludwig->lb, ludwig->wall,
			   ludwig->collinfo);
      wall_bbl(ludwig->wall);
      TIMER_stop(TIMER_BBL);
    }
    else {
      /* No hydrodynamics, but update colloids in response to
       * external forces. */

      bbl_update_colloids(ludwig->bbl, ludwig->wall, ludwig->collinfo);
    }




    /* There must be no halo updates between bounce back
     * and propagation, as the halo regions are active */

    if (ludwig->hydro) {
      TIMER_start(TIMER_PROPAGATE);
      lb_propagation(ludwig->lb);
      TIMER_stop(TIMER_PROPAGATE);
    }

    TIMER_start(TIMER_DIAGNOSTIC_OUTPUT); /* Time diagnostics and i/o */

    /* Configuration dump */

    if (is_config_step()) {
      io_event_t event = {0};
      pe_info(ludwig->pe, "Writing distribution output at step %d!\n", step);
      lb_memcpy(ludwig->lb, tdpMemcpyDeviceToHost);
      lb_io_write(ludwig->lb, step, &event);
    }

    /* is_measurement_step() is here to prevent 'breaking' old input
     * files; it should really be removed. */

    if (is_config_step() || is_measurement_step() || is_colloid_io_step()) {
      if (ncolloid > 0) {
	pe_info(ludwig->pe, "Writing colloid output at step %d!\n", step);
	sprintf(filename, "%s%8.8d", "config.cds", step);
	colloid_io_write(ludwig->cio, filename);
      }
    }

    if (is_phi_output_step() || is_config_step()) {

      if (ludwig->phi) {
	io_event_t event = {0};
	pe_info(ludwig->pe, "Writing phi file at step %d!\n", step);
	field_memcpy(ludwig->phi, tdpMemcpyDeviceToHost);
	field_io_write(ludwig->phi, step, &event);
      }

      if (ludwig->p) {
	io_event_t event = {0};
	pe_info(ludwig->pe, "Writing p file at step %d!\n", step);
	field_memcpy(ludwig->p, tdpMemcpyDeviceToHost);
	field_io_write(ludwig->p, step, &event);
      }

      if (ludwig->q) {
	io_event_t event = {0};
	pe_info(ludwig->pe, "Writing q file at step %d!\n", step);
	field_memcpy(ludwig->q, tdpMemcpyDeviceToHost);
	io_replace_values(ludwig->q, ludwig->map, MAP_COLLOID, 0.00001);
	field_io_write(ludwig->q, step, &event);
      }
    }

    if (ludwig->psi) {
      if (is_psi_output_step() || is_config_step()) {
	pe_info(ludwig->pe, "Writing psi file at step %d!\n", step);
	psi_io_write(ludwig->psi, step);
      }
    }

    /* Measurements */

    if (is_measurement_step()) {
      /* TODO: Allow calibration to be taken its own measurement frequency */
      stats_sigma_measure(ludwig->stat_sigma, step);
    }

    if (is_shear_measurement_step()) {
      lb_memcpy(ludwig->lb, tdpMemcpyDeviceToDevice);
      stats_rheology_stress_profile_accumulate(ludwig->stat_rheo, ludwig->lb,
					       ludwig->fe, ludwig->hydro);
    }

    if (is_shear_output_step()) {
      sprintf(filename, "str-%8.8d.dat", step);
      stats_rheology_stress_section(ludwig->stat_rheo, filename);
      stats_rheology_stress_profile_zero(ludwig->stat_rheo);
    }

    if (ludwig->hydro && (is_vel_output_step() || is_config_step())) {
      io_event_t event = {0};
      pe_info(ludwig->pe, "Writing rho/velocity output at step %d!\n", step);
      hydro_memcpy(ludwig->hydro, tdpMemcpyDeviceToHost);
      hydro_io_write(ludwig->hydro, step, &event);
    }

    /* Print progress report */

    timekeeper_step(&ludwig->tk);

    if (is_statistics_step()) {

      ludwig_report_statistics(ludwig, step);
      ludwig_report_momentum(ludwig);

      if (ludwig->hydro) {
	wall_is_pm(ludwig->wall, &is_pm);
	hydro_memcpy(ludwig->hydro, tdpMemcpyDeviceToHost);
	stats_velocity_minmax(&statvel, ludwig->hydro, ludwig->map);
      }

      lb_collision_stats_kt(ludwig->lb, ludwig->map);

      pe_info(ludwig->pe, "\nCompleted cycle %d\n", step);
    }

    stats_ahydro_accumulate(ludwig->stat_ah, step);

    TIMER_stop(TIMER_DIAGNOSTIC_OUTPUT);
    TIMER_stop(TIMER_STEPS); /* inclusive of diagnostic/io */

    /* Next time step */
  }


  /* End of time step loop. A barrier, before closing down. */
  MPI_Barrier(comm);

  /* Shut down cleanly. Give the timer statistics. Finalise PE. */

  if (ludwig->poisson) ludwig->poisson->impl->free(&ludwig->poisson);
  if (ludwig->psi) psi_free(&ludwig->psi);

  if (ludwig->stat_rheo) stats_rheology_free(ludwig->stat_rheo);
  if (ludwig->stat_turb) stats_turbulent_free(ludwig->stat_turb);
  if (ludwig->stat_ah)   stats_ahydro_free(ludwig->stat_ah);

  if (ludwig->phi_grad) field_grad_free(ludwig->phi_grad);
  if (ludwig->p_grad)   field_grad_free(ludwig->p_grad);
  if (ludwig->q_grad)   field_grad_free(ludwig->q_grad);
  if (ludwig->phi)      field_free(ludwig->phi);
  if (ludwig->p)        field_free(ludwig->p);
  if (ludwig->q)        field_free(ludwig->q);

  bbl_free(ludwig->bbl);
  colloids_info_free(ludwig->collinfo);

  if (ludwig->inflow) ludwig->inflow->func->free(ludwig->inflow);
  if (ludwig->outflow) ludwig->outflow->func->free(ludwig->outflow);
  if (ludwig->phi_inflow) ludwig->phi_inflow->func->free(ludwig->phi_inflow);
  if (ludwig->phi_outflow) ludwig->phi_outflow->func->free(ludwig->phi_outflow);

  if (ludwig->interact) interact_free(ludwig->interact);
  if (ludwig->cio)      colloid_io_free(ludwig->cio);

  if (ludwig->wall)      wall_free(ludwig->wall);
#ifdef OLD_SHIT
  if (ludwig->noise_phi) noise_free(ludwig->noise_phi);
  if (ludwig->noise_rho) noise_free(ludwig->noise_rho);
#else
  if (ludwig->noise)     noise_free(&ludwig->noise);
#endif
  if (ludwig->be)        beris_edw_free(ludwig->be);
  if (ludwig->map)       map_free(&ludwig->map);
  if (ludwig->pch)       phi_ch_free(ludwig->pch);
  if (ludwig->pth)       pth_free(ludwig->pth);
  if (ludwig->hydro)     hydro_free(ludwig->hydro);
  if (ludwig->lb)        lb_free(ludwig->lb);

  if (ludwig->stat_sigma) stats_sigma_free(ludwig->stat_sigma);
  if (ludwig->fe) ludwig->fe->func->free(ludwig->fe);

  TIMER_stop(TIMER_TOTAL);
  TIMER_statistics();

  physics_free(ludwig->phys);
  if (ludwig->le) lees_edw_free(ludwig->le);
  cs_free(ludwig->cs);
  rt_report_unused_keys(ludwig->rt, RT_INFO);
  rt_free(ludwig->rt);
  pe_free(ludwig->pe);

  free(ludwig);

  return;
}

/*****************************************************************************
 *
 *  ludwig_report_momentum
 *
 *  Tidy report of the current momentum of the system.
 *
 *****************************************************************************/

static int ludwig_report_momentum(ludwig_t * ludwig) {

  int n;
  int ncolloid;
  int is_pm;

  double g[3];         /* Fluid momentum (total) */
  double gc[3];        /* Colloid momentum (total) */
  double gwall[3];     /* Wall momentum (for accounting purposes only) */
  double gtotal[3];

  MPI_Comm comm;
  pe_t * pe = NULL;

  pe = ludwig->pe;
  pe_mpi_comm(pe, &comm);
  wall_is_pm(ludwig->wall, &is_pm);

  for (n = 0; n < 3; n++) {
    gtotal[n] = 0.0;
    g[n] = 0.0;
    gc[n] = 0.0;
    gwall[n] = 0.0;
  }

  stats_distribution_momentum(ludwig->lb, ludwig->map, g);
  stats_colloid_momentum(ludwig->collinfo, gc);
  colloids_info_ntotal(ludwig->collinfo, &ncolloid);

  if (wall_present(ludwig->wall) || is_pm) {
    double gtmp[3];
    wall_momentum(ludwig->wall, gtmp);
    MPI_Reduce(gtmp, gwall, 3, MPI_DOUBLE, MPI_SUM, 0, comm);
  }

  for (n = 0; n < 3; n++) {
    gtotal[n] = g[n] + gc[n] + gwall[n];
  }

  pe_info(pe, "\n");
  pe_info(pe, "Momentum - x y z\n");
  pe_info(pe, "[total   ] %14.7e %14.7e %14.7e\n", gtotal[X], gtotal[Y], gtotal[Z]);
  pe_info(pe, "[fluid   ] %14.7e %14.7e %14.7e\n", g[X], g[Y], g[Z]);
  if (ncolloid > 0) {
    pe_info(pe, "[colloids] %14.7e %14.7e %14.7e\n", gc[X], gc[Y], gc[Z]);
  }
  if (wall_present(ludwig->wall) || is_pm) {
    pe_info(pe, "[walls   ] %14.7e %14.7e %14.7e\n", gwall[X], gwall[Y], gwall[Z]);
  }

  return 0;
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
 *  This is currently rather repetitive, so some rationalisation
 *  is required.
 *
 *****************************************************************************/

int free_energy_init_rt(ludwig_t * ludwig) {

  int n = 0;
  int p;
  int nf;
  int nk;
  int ngrad;
  int nhalo;
  double value;
  char description[BUFSIZ];

  pe_t * pe = NULL;
  rt_t * rt = NULL;
  cs_t * cs = NULL;
  lees_edw_t * le = NULL;

  lees_edw_options_t le_info = {0};
  lees_edw_options_t * info = &le_info;

  assert(ludwig);
  assert(ludwig->pe);
  assert(ludwig->rt);

  pe = ludwig->pe;
  rt = ludwig->rt;
  cs_create(pe,&cs);

  lees_edw_init_rt(rt, info);

  n = rt_string_parameter(rt, "free_energy", description, BUFSIZ);

  if (strcmp(description, "none") == 0) {
    /* Appropriate for single fluid */
    pe_info(pe, "\n");
    pe_info(pe, "No free energy selected\n");

    nhalo = 1;
    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);
    pth_create(pe, cs, FE_FORCE_METHOD_NO_FORCE, &ludwig->pth);
  }
  else if (strcmp(description, "symmetric") == 0) {

    phi_ch_info_t ch_options = {0};
    fe_symm_t * fe = NULL;

    /* Symmetric free energy via finite difference */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 2;   /* Require stress divergence. */
    ngrad = 2;   /* \nabla^2 required */

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);
      io_info_args_rt(rt, RT_FATAL, "phi", IO_INFO_READ_WRITE, &opts.iodata);
      field_create(pe, cs, le, "phi", &opts, &ludwig->phi);
      field_grad_create(pe, ludwig->phi, ngrad, &ludwig->phi_grad);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");
    fe_symm_create(pe, cs, ludwig->phi, ludwig->phi_grad, &fe);
    fe_symmetric_init_rt(pe, rt, fe);

    pe_info(pe, "\n");
    pe_info(pe, "Using Cahn-Hilliard finite difference solver.\n");

    rt_key_required(rt, "mobility", RT_FATAL);
    rt_double_parameter(rt, "mobility", &value);
    physics_mobility_set(ludwig->phys, value);
    pe_info(pe, "Mobility M            = %12.5e\n", value);

    pe_info(pe, "Order parameter noise = %3s\n", "off");

    rt_int_parameter(rt, "cahn_hilliard_options_conserve",
		     &ch_options.conserve);
    phi_ch_create(pe, cs, le, &ch_options, &ludwig->pch);

    /* Force */

    {
      fe_force_method_enum_t method = fe_force_method_default();

      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are supported */
      switch (method) {
      case FE_FORCE_METHOD_NO_FORCE:
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
      case FE_FORCE_METHOD_PHI_GRADMU:
      case FE_FORCE_METHOD_PHI_GRADMU_CORRECTION:
	break;
      case FE_FORCE_METHOD_RELAXATION_SYMM:
	fe->super.use_stress_relaxation = 1;
	break;
      default:
	pe_fatal(pe, "symmetric free energy force_method not available\n");
      }

      pth_create(pe, cs, method, &ludwig->pth);
      pe_info(pe, "Force calculation:      %s\n",
	      fe_force_method_to_string(method));
    }

    ludwig->fe_symm = fe;
    ludwig->fe = (fe_t *) fe;
  }
  else if (strcmp(description, "symmetric_noise") == 0) {

    phi_ch_info_t ch_options = {0};
    fe_symm_t * fe = NULL;

    /* Symmetric via finite difference plus isothermal fluctuations */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 3;   /* Additional point cf no noise */
    ngrad = 2;   /* \nabla^2 required */

    ch_options.noise = 1;  /* "symmetric_noise" => fluctuations */

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);
      io_info_args_rt(rt, RT_FATAL, "phi", IO_INFO_READ_WRITE, &opts.iodata);
      field_create(pe, cs, le, "phi", &opts, &ludwig->phi);
      field_grad_create(pe, ludwig->phi, ngrad, &ludwig->phi_grad);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");
    fe_symm_create(pe, cs, ludwig->phi, ludwig->phi_grad, &fe);
    fe_symmetric_init_rt(pe, rt, fe);

    pe_info(pe, "\n");
    pe_info(pe, "Using Cahn-Hilliard finite difference solver.\n");

    rt_key_required(rt, "mobility", RT_FATAL);
    rt_double_parameter(rt, "mobility", &value);
    physics_mobility_set(ludwig->phys, value);
    pe_info(pe, "Mobility M            = %12.5e\n", value);

    rt_int_parameter(rt, "cahn_hilliard_options_conserve",
		     &ch_options.conserve);
    phi_ch_create(pe, cs, le, &ch_options, &ludwig->pch);

    /* Order parameter noise */

    pe_info(pe, "Order parameter noise = %3s\n", (ch_options.noise == 0) ? "off" : " on");

    /* Force */

    {
      fe_force_method_enum_t method = fe_force_method_default();

      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are supported */
      switch (method) {
      case FE_FORCE_METHOD_NO_FORCE:
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
      case FE_FORCE_METHOD_PHI_GRADMU:
      case FE_FORCE_METHOD_PHI_GRADMU_CORRECTION:
	break;
      case FE_FORCE_METHOD_RELAXATION_SYMM:
	fe->super.use_stress_relaxation = 1;
	break;
      default:
	pe_fatal(pe, "symmetric free energy force_method not available\n");
      }

      pth_create(pe, cs, method, &ludwig->pth);
      pe_info(pe, "Force calculation:      %s\n",
	      fe_force_method_to_string(method));
    }

    ludwig->fe_symm = fe;
    ludwig->fe = (fe_t *) fe;
  }
  else if (strcmp(description, "symmetric_lb") == 0) {

    fe_symm_t * fe = NULL;

    /* Symmetric free energy via full lattice kintic equation */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 1;   /* Require one point for LB. */
    ngrad = 2;   /* \nabla^2 required */

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);
      io_info_args_rt(rt, RT_FATAL, "phi", IO_INFO_READ_WRITE, &opts.iodata);
      field_create(pe, cs, le, "phi", &opts, &ludwig->phi);
      field_grad_create(pe, ludwig->phi, ngrad, &ludwig->phi_grad);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");
    fe_symm_create(pe, cs, ludwig->phi, ludwig->phi_grad, &fe);
    fe_symmetric_init_rt(pe, rt, fe);

    pe_info(pe, "\n");
    pe_info(pe, "Using full lattice Boltzmann solver for Cahn-Hilliard:\n");

    rt_key_required(rt, "mobility", RT_FATAL);
    rt_double_parameter(rt, "mobility", &value);
    physics_mobility_set(ludwig->phys, value);
    pe_info(pe, "Mobility M            = %12.5e\n", value);

    /* No explicit force is relevant */
    fe_force_method_rt_messages(rt, RT_INFO);
    pth_create(pe, cs, FE_FORCE_METHOD_NO_FORCE, &ludwig->pth);

    ludwig->fe_symm = fe;
    ludwig->fe = (fe_t *) fe;
  }
  else if (strcmp(description, "brazovskii") == 0) {

    /* Brazovskii (always finite difference). */

    phi_ch_info_t ch_options = {0};
    fe_brazovskii_t * fe = NULL;
    nf = 1;      /* 1 scalar order parameter */
    nhalo = 3;   /* Required for stress diveregnce. */
    ngrad = 4;   /* (\nabla^2)^2 required */

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);
      io_info_args_rt(rt, RT_FATAL, "phi", IO_INFO_READ_WRITE, &opts.iodata);
      field_create(pe, cs, le, "phi", &opts, &ludwig->phi);
      field_grad_create(pe, ludwig->phi, ngrad, &ludwig->phi_grad);
      phi_ch_create(pe, cs, le, &ch_options, &ludwig->pch);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");
    fe_brazovskii_create(pe, cs, ludwig->phi, ludwig->phi_grad, &fe);
    fe_brazovskii_init_rt(pe, rt, fe);

    pe_info(pe, "\n");
    pe_info(pe, "Using Cahn-Hilliard solver:\n");

    rt_key_required(rt, "mobility", RT_FATAL);
    rt_double_parameter(rt, "mobility", &value);
    physics_mobility_set(ludwig->phys, value);
    pe_info(pe, "Mobility M            = %12.5e\n", value);

    {
      fe_force_method_enum_t method = fe_force_method_default();

      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are supported */
      switch (method) {
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
      case FE_FORCE_METHOD_PHI_GRADMU:
      case FE_FORCE_METHOD_PHI_GRADMU_CORRECTION:
	break;
      default:
	pe_fatal(pe, "brazovskii: force_method not available\n");
      }

      pth_create(pe, cs, method, &ludwig->pth);
      pe_info(pe, "Force calculation:      %s\n",
	      fe_force_method_to_string(method));
    }

    ludwig->fe_braz = fe;
    ludwig->fe = (fe_t *) fe;
  }
  else if (strcmp(description, "surfactant") == 0) {

    fe_surf_param_t param;
    ch_info_t options = {0};
    fe_surf_t * fe = NULL;

    nf = 2;       /* Composition, surfactant: "phi" and "psi" */
    nhalo = 2;
    ngrad = 2;

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);

    /* No Lees Edwards for the time being */

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);
      field_create(pe, cs, NULL, "surfactant1", &opts, &ludwig->phi);
    }

    field_grad_create(pe, ludwig->phi, ngrad, &ludwig->phi_grad);

    pe_info(pe, "\n");
    pe_info(pe, "Surfactant free energy\n");
    pe_info(pe, "----------------------\n");

    fe_surf_param_rt(pe, rt, &param);
    fe_surf_create(pe, cs, ludwig->phi, ludwig->phi_grad, param, &fe);
    fe_surf_info(fe);

    /* Cahn Hilliard */

    options.nfield = nf;

    n = rt_double_parameter(rt, "surf_mobility_phi", &options.mobility[0]);
    if (n == 0) pe_fatal(pe, "Please set mobility_phi in the input\n");

    n = rt_double_parameter(rt, "surf_mobility_psi", &options.mobility[1]);
    if (n == 0) pe_fatal(pe, "Please set mobility_psi in the input\n");

    ch_create(pe, cs, options, &ludwig->ch);

    pe_info(pe, "\n");
    pe_info(pe, "Using Cahn-Hilliard solver:\n");
    ch_info(ludwig->ch);

    /* Coupling between momentum and free energy */
    /* Hydrodynamics sector (move to hydro_rt?) */
    {
      fe_force_method_enum_t method = fe_force_method_default();

      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are possible (if not tested) */
      switch (method) {
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
      case FE_FORCE_METHOD_PHI_GRADMU:
      case FE_FORCE_METHOD_PHI_GRADMU_CORRECTION:
	break;
      case FE_FORCE_METHOD_RELAXATION_SYMM:
	fe->super.use_stress_relaxation = 1;
	break;
      default:
	pe_fatal(pe, "surfactant free energy force_method not available\n");
      }

      pth_create(pe, cs, method, &ludwig->pth);
    }

    ludwig->fe_surf = fe;
    ludwig->fe = (fe_t *) fe;

  }
  else if (strcmp(description, "ternary") == 0) {

    fe_ternary_param_t param = {0};
    ch_info_t options = {0};
    fe_ternary_t * fe = NULL;

    nf = 2;       /* Composition, ternary: "phi" and "psi" */
    nhalo = 2;
    ngrad = 2;

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);

    /* No Lees Edwards for the time being */

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);
      io_info_args_rt(rt, RT_FATAL, "phi", IO_INFO_READ_WRITE, &opts.iodata);
      field_create(pe, cs, NULL, "phi", &opts, &ludwig->phi);
    }

    field_grad_create(pe, ludwig->phi, ngrad, &ludwig->phi_grad);

    pe_info(pe, "\n");
    pe_info(pe, "Ternary free energy\n");
    pe_info(pe, "----------------------\n");

    fe_ternary_param_rt(pe, rt, &param);
    fe_ternary_create(pe, cs, ludwig->phi, ludwig->phi_grad, param, &fe);
    fe_ternary_info(fe);

    /* Allow either possibility for gradient computation... */
    grad_2d_ternary_solid_fe_set(fe);
    grad_3d_ternary_solid_fe_set(fe);

    /* Cahn Hilliard */

    options.nfield = nf;

    n = rt_double_parameter(rt, "ternary_mobility_phi", &options.mobility[0]);
    if (n == 0) pe_fatal(pe, "Please set ternary_mobility_phi in the input\n");

    n = rt_double_parameter(rt, "ternary_mobility_psi", &options.mobility[1]);
    if (n == 0) pe_fatal(pe, "Please set ternary_mobility_psi in the input\n");

    ch_create(pe, cs, options, &ludwig->ch);

    pe_info(pe, "\n");
    pe_info(pe, "Using Cahn-Hilliard solver:\n");
    ch_info(ludwig->ch);

    /* Coupling between momentum and free energy */
    /* Default method for ternary free energy: phi_gradmu */
    {
      fe_force_method_enum_t method = FE_FORCE_METHOD_PHI_GRADMU;

      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are supported */
      switch (method) {
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
      case FE_FORCE_METHOD_PHI_GRADMU:
      case FE_FORCE_METHOD_PHI_GRADMU_CORRECTION:
	break;
      case FE_FORCE_METHOD_RELAXATION_SYMM:
	fe->super.use_stress_relaxation = 1;
	break;
      default:
	pe_fatal(pe, "ternary free energy: force_method not available\n");
      }

      pth_create(pe, cs, method, &ludwig->pth);
      pe_info(pe, "Force calculation:      %s\n",
	      fe_force_method_to_string(method));
    }

    ludwig->fe_ternary = fe;
    ludwig->fe = (fe_t *) fe;
  }
  else if (strcmp(description, "lc_blue_phase") == 0) {

    fe_lc_t * fe = NULL;

    /* Liquid crystal (always finite difference). */

    nf = NQAB;   /* Tensor order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);

      if (rt_switch(rt, "field_halo_openmp")) {
	opts.haloscheme = FIELD_HALO_OPENMP;
	opts.haloverbose = rt_switch(rt, "field_halo_verbose");
      }
      if (rt_switch(rt, "field_data_use_first_touch")) {
	opts.usefirsttouch = 1;
      }
      io_info_args_rt(rt, RT_FATAL, "q", IO_INFO_READ_WRITE, &opts.iodata);

      field_create(pe, cs, le, "q", &opts, &ludwig->q);
      field_grad_create(pe, ludwig->q, ngrad, &ludwig->q_grad);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");

    fe_lc_create(pe, cs, le, ludwig->q, ludwig->q_grad, &fe);
    beris_edw_create(pe, cs, le, &ludwig->be);
    blue_phase_init_rt(pe, rt, fe, ludwig->be);

    {
      fe_force_method_enum_t method = fe_force_method_default();

      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are supported */
      switch (method) {
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
	break;
      case FE_FORCE_METHOD_RELAXATION_ANTI:
	fe->super.use_stress_relaxation = 1;
	tdpAssert(tdpMemcpy(&fe->target->super.use_stress_relaxation,
			    &fe->super.use_stress_relaxation, sizeof(int),
			    tdpMemcpyHostToDevice));
	break;
      default:
	pe_fatal(pe, "liquid crystal: force_method not available\n");
      }

      pth_create(pe, cs, method, &ludwig->pth);
    }

    /* Assign anchoring information (both gradient possibilities) */
    grad_lc_anch_create(pe, cs, NULL, NULL, NULL, fe, NULL);
    grad_s7_anchoring_create(pe, cs, NULL, fe, NULL);

    ludwig->fe_lc = fe;
    ludwig->fe = (fe_t *) fe;
  }
  else if (strcmp(description, "polar_active") == 0) {

    /* Polar active. */
    fe_polar_t * fe = NULL;
    leslie_param_t lep = {0};

    nf = NVECTOR;/* Vector order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);
      io_info_args_rt(rt, RT_FATAL, "p", IO_INFO_READ_WRITE, &opts.iodata);
      field_create(pe, cs, le, "p", &opts, &ludwig->p);
      field_grad_create(pe, ludwig->p, ngrad, &ludwig->p_grad);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");

    fe_polar_create(pe, cs, ludwig->p, ludwig->p_grad, &fe);
    polar_active_run_time(pe, rt, fe);
    ludwig->fe = (fe_t *) fe;

    rt_double_parameter(rt, "leslie_ericksen_gamma", &lep.Gamma);
    rt_double_parameter(rt, "leslie_ericksen_swim",  &lep.swim);
    rt_double_parameter(rt, "polar_active_lambda",   &lep.lambda);

    pe_info(pe, "Rotational diffusion     = %12.5e\n", lep.Gamma);
    pe_info(pe, "Self-advection parameter = %12.5e\n", lep.swim);

    pth_create(pe, cs, FE_FORCE_METHOD_STRESS_DIVERGENCE, &ludwig->pth);
    leslie_ericksen_create(pe, cs, fe, ludwig->p, &lep, &ludwig->leslie);
  }
  else if(strcmp(description, "lc_droplet") == 0) {

    phi_ch_info_t ch_options = {0};
    fe_symm_t * symm = NULL;
    fe_lc_t * lc = NULL;
    fe_lc_droplet_t * fe = NULL;

    /* liquid crystal droplet */
    pe_info(pe, "\n");
    pe_info(pe, "Liquid crystal droplet free energy selected\n");

    /* first do the symmetric */
    nf = 1;      /* 1 scalar order parameter */
    nhalo = 2;   /* Require stress divergence. */
    ngrad = 3;   /* \nabla^2 and d_a d_b required */

    /* Noise requires additional stencil point for Cahn Hilliard */
    if (strcmp(description, "symmetric_noise") == 0) {
      nhalo = 3;
    }

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);

      if (rt_switch(rt, "field_halo_openmp")) {
	opts.haloscheme = FIELD_HALO_OPENMP;
	opts.haloverbose = rt_switch(rt, "field_halo_verbose");
      }
      if (rt_switch(rt, "field_data_use_first_touch")) {
	opts.usefirsttouch = 1;
      }
      io_info_args_rt(rt, RT_FATAL, "phi", IO_INFO_READ_WRITE, &opts.iodata);

      field_create(pe, cs, le, "phi", &opts, &ludwig->phi);
      field_grad_create(pe, ludwig->phi, ngrad, &ludwig->phi_grad);
      phi_ch_create(pe, cs, le, &ch_options, &ludwig->pch);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");
    fe_symm_create(pe, cs, ludwig->phi, ludwig->phi_grad, &symm);
    fe_symmetric_init_rt(pe, rt, symm);

    pe_info(pe, "\n");
    pe_info(pe, "Using Cahn-Hilliard finite difference solver.\n");

    rt_key_required(rt, "mobility", RT_FATAL);
    rt_double_parameter(rt, "mobility", &value);
    physics_mobility_set(ludwig->phys, value);
    pe_info(pe, "Mobility M            = %12.5e\n", value);

    /* Liquid crystal part */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    {
      field_options_t opts = field_options_ndata_nhalo(NQAB, nhalo);

      if (rt_switch(rt, "field_halo_openmp")) {
	opts.haloscheme = FIELD_HALO_OPENMP;
	opts.haloverbose = rt_switch(rt, "field_halo_verbose");
      }
      if (rt_switch(rt, "field_data_use_first_touch")) {
	opts.usefirsttouch = 1;
      }
      io_info_args_rt(rt, RT_FATAL, "q", IO_INFO_READ_WRITE, &opts.iodata);
      field_create(pe, cs, le, "q", &opts, &ludwig->q);
      field_grad_create(pe, ludwig->q, ngrad, &ludwig->q_grad);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");

    fe_lc_create(pe, cs, le, ludwig->q, ludwig->q_grad, &lc);
    beris_edw_create(pe, cs, le, &ludwig->be);
    blue_phase_init_rt(pe, rt, lc, ludwig->be);

    fe_lc_droplet_create(pe, cs, lc, symm, &fe);
    fe_lc_droplet_run_time(pe, rt, fe);

    /* Force */

    {
      fe_force_method_enum_t method = fe_force_method_default();

      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are supported */
      switch (method) {
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
	break;
      case FE_FORCE_METHOD_RELAXATION_ANTI:
	fe->super.use_stress_relaxation = 1;
	tdpAssert(tdpMemcpy(&fe->target->super.use_stress_relaxation,
			    &fe->super.use_stress_relaxation, sizeof(int),
			    tdpMemcpyHostToDevice));
	break;
      default:
	pe_fatal(pe, "liquid crystal droplet: force_method not available\n");
      }

      pth_create(pe, cs, method, &ludwig->pth);

      pe_info(pe, "\n");
      pe_info(pe, "Coupled free energy\n");
      pe_info(pe, "Force calculation:      %s\n",
	      fe_force_method_to_string(method));
    }

    p = rt_switch(rt, "lc_noise");
    if (p) pe_fatal(pe, "Not accepting noise in lc droplet until tested\n");

    grad_lc_anch_create(pe, cs, NULL, ludwig->phi, NULL, lc, NULL);

    ludwig->fe_symm = symm;
    ludwig->fe_lc = lc;
    ludwig->fe = (fe_t *) fe;
  }
  else if(strcmp(description, "fe_electro") == 0) {

    int ifail = 0;
    fe_electro_t * fe = NULL;
    fe_force_method_enum_t method = fe_force_method_default();
    int psi_method = PSI_FORCE_NONE;


    nk = 2;    /* Number of charge densities always 2 for now */
    nhalo = 2; /* Default to stress divergence (see force). */

    /* Single fluid electrokinetic free energy */

    /* Force */

    {
      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are supported */
      switch (method) {
      case FE_FORCE_METHOD_PHI_GRADMU_CORRECTION:
	nhalo = 1;
	psi_method = PSI_FORCE_GRADMU;
	break;
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
	nhalo = 2;
	psi_method = PSI_FORCE_DIVERGENCE;
	break;
      default:
	pe_fatal(pe, "electrokinetic: force_method not available\n");
      }
    }

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    pe_info(pe, "\n");
    pe_info(pe, "Free energy details\n");
    pe_info(pe, "-------------------\n\n");
    pe_info(pe, "Electrokinetics (single fluid) selected\n");

    pe_info(pe, "\n");
    pe_info(pe, "Parameters:\n");

    {
      /* Options here currently include solver options ... */
      psi_options_t opts = psi_options_default(nhalo);
      psi_options_rt(pe, cs, rt, &opts);
      psi_create(pe, cs, &opts, &ludwig->psi);
      psi_force_method_set(ludwig->psi, psi_method);
      psi_info(pe, ludwig->psi);
    }

    pe_info(pe, "Force calculation:      %s\n",
	    fe_force_method_to_string(method));

    /* Create FE objects and set function pointers */
    fe_electro_create(pe, ludwig->psi, &fe);
    ludwig->fe = (fe_t *) fe;

    /* Uniform solver ok */

    ifail = psi_solver_create(ludwig->psi, &ludwig->poisson);
    if (ifail != 0) {
      pe_info(pe, "Poisson solver initialisation failed\n");
      pe_info(pe, "This probably means you specified \"petsc\" but it has\n");
      pe_info(pe, "not been compiled. Please specify sor in the input.\n");
      pe_fatal(pe, "Please check and try again\n");
    }
  }
  else if(strcmp(description, "fe_electro_symmetric") == 0) {

    phi_ch_info_t ch_options = {0};
    fe_symm_t * fe_symm = NULL;
    fe_electro_t * fe_elec = NULL;
    fe_es_t * fes = NULL;
    double e1, e2;
    double mu[2];
    double lbjerrum2;

    /* Binary fluid plus electrokinetics */

    nf = 1;      /* Single scalar order parameter phi */
    nk = 2;      /* Two charge species */
    nhalo = 2;   /* Require stress divergence. */
    ngrad = 2;   /* \nabla^2 phi */


    /* First, the symmetric part. */

    cs_nhalo_set(cs, nhalo);
    coords_init_rt(pe, rt, cs);
    lees_edw_create(pe, cs, info, &le);
    lees_edw_info(le);

    {
      field_options_t opts = field_options_ndata_nhalo(nf, nhalo);
      field_create(pe, cs, le, "phi", &opts, &ludwig->phi);
      field_grad_create(pe, ludwig->phi, ngrad, &ludwig->phi_grad);
      phi_ch_create(pe, cs, le, &ch_options, &ludwig->pch);
    }

    pe_info(pe, "\n");
    pe_info(pe, "Charged binary fluid 'Electrosymmetric' free energy\n");
    pe_info(pe, "---------------------------------------------------\n");

    pe_info(pe, "\n");
    pe_info(pe, "Symmetric part\n");
    pe_info(pe, "--------------\n\n");

    fe_symm_create(pe, cs, ludwig->phi, ludwig->phi_grad, &fe_symm);
    fe_symmetric_init_rt(pe, rt, fe_symm);

    pe_info(pe, "\n");
    pe_info(pe, "Using Cahn-Hilliard finite difference solver.\n");

    rt_key_required(rt, "mobility", RT_FATAL);
    rt_double_parameter(rt, "mobility", &value);
    physics_mobility_set(ludwig->phys, value);
    pe_info(pe, "Mobility M            = %12.5e\n", value);

    /* Electrokinetic part */

    pe_info(pe, "\n");
    pe_info(pe, "Electrokinetic part\n");
    pe_info(pe, "-------------------\n\n");

    pe_info(pe, "Parameters:\n");

    {
      psi_options_t opts = psi_options_default(nhalo);
      psi_options_rt(pe, cs, rt, &opts);
      psi_create(pe, cs, &opts, &ludwig->psi);
      psi_info(pe, ludwig->psi);
      psi_bjerrum_length2(&opts, &lbjerrum2);
    }
    fe_electro_create(pe, ludwig->psi, &fe_elec);

    /* Coupling part */

    pe_info(pe, "\n");
    pe_info(pe, "Coupling part\n");
    pe_info(pe, "-------------\n");

    /* Create FE objects and set function pointers */
    fe_es_create(pe, cs, fe_symm, fe_elec, ludwig->psi, &fes);

    /* Dielectric contrast */

    /* Call permittivities, e1 == e2 has been set as default */
    psi_epsilon(ludwig->psi, &e1);
    psi_epsilon2(ludwig->psi, &e2);

    /* Solvation free energy difference: nk = 2 */

    mu[0] = 0.0;
    mu[1] = 0.0;

    rt_double_parameter(rt, "electrosymmetric_delta_mu0", mu);
    rt_double_parameter(rt, "electrosymmetric_delta_mu1", mu + 1);

    fe_es_deltamu_set(fes, nk, mu);

    pe_info(pe, "Second permittivity:      %15.7e\n", e2);
    pe_info(pe, "Dielectric average:       %15.7e\n", 0.5*(e1 + e2));
    pe_info(pe, "Dielectric contrast:      %15.7e\n", (e1-e2)/(e1+e2));
    pe_info(pe, "Second Bjerrum length:    %15.7e\n", lbjerrum2);
    pe_info(pe, "Solvation dmu species 0:  %15.7e\n", mu[0]);
    pe_info(pe, "Solvation dmu species 1:  %15.7e\n", mu[1]);

    /* f_vare_t function */
    /* If permittivities really not the same number... */

    if (util_double_same(e1, e2)) {
      int ifail = 0;
      ifail = psi_solver_create(ludwig->psi, &ludwig->poisson);
      if (ifail != 0) {
	pe_info(pe, "Poisson solver initialisation failed\n");
	pe_info(pe, "This may mean you specified \"petsc\" but it has\n");
	pe_info(pe, "not been compiled. Please specify sor in the input.\n");
	pe_fatal(pe, "Please check and try again\n");
      }
      pe_info(pe, "Poisson solver:           %15s\n", "uniform");
    }
    else {
      int ifail = 0;
      var_epsilon_t user = {.fe = (fe_t *) fes,
			    .epsilon = (var_epsilon_ft) fe_es_var_epsilon};
      ifail = psi_solver_var_epsilon_create(ludwig->psi, user, &ludwig->poisson);
      if (ifail != 0) {
	pe_info(pe, "Poisson solver initialisation failed\n");
	pe_info(pe, "This may mean you specified \"petsc\" but it has\n");
	pe_info(pe, "not been compiled. Please specify sor in the input.\n");
	pe_fatal(pe, "Please check and try again\n");
      }
      pe_info(pe, "Poisson solver:           %15s\n", "heterogeneous");
    }

    /* Force */

    {
      fe_force_method_enum_t method = fe_force_method_default();

      fe_force_method_rt_messages(rt, RT_INFO);
      fe_force_method_rt(rt, RT_FATAL, &method);

      /* The following are supported */
      switch (method) {
      case FE_FORCE_METHOD_PHI_GRADMU_CORRECTION:
	psi_force_method_set(ludwig->psi, PSI_FORCE_GRADMU);
	break;
      case FE_FORCE_METHOD_STRESS_DIVERGENCE:
	psi_force_method_set(ludwig->psi, PSI_FORCE_DIVERGENCE);
	break;
      default:
	pe_fatal(pe, "electrosymmetric: force_method not available\n");
      }

      pe_info(pe, "\n");
      pe_info(pe, "Coupled free energy\n");
      pe_info(pe, "Force calculation:      %s\n",
	      fe_force_method_to_string(method));
    }

    ludwig->fe_symm = fe_symm;
    ludwig->fe = (fe_t *) fes;

  }
  else {
    if (n == 1) {
      /* The user has put something which hasn't been recognised,
       * suggesting a spelling mistake */
      pe_info(pe, "free_energy %s not recognised.\n", description);
      pe_fatal(pe, "Please check and try again.\n");
    }
  }

  ludwig->cs = cs;
  ludwig->le = le;

  return 0;
}

/*****************************************************************************
 *
 *  visc_model_init_rt
 *
 *****************************************************************************/

int visc_model_init_rt(pe_t * pe, rt_t * rt, ludwig_t * ludwig) {

  int key;
  char description[BUFSIZ/2];

  assert(pe);
  assert(rt);
  assert(ludwig);

  key = rt_string_parameter(rt, "viscosity_model", description, BUFSIZ/2);

  if (strcmp(description, "arrhenius") == 0) {
    cs_t * cs = ludwig->cs;
    field_t * phi = ludwig->phi;
    visc_arrhenius_param_t param = {0};
    visc_arrhenius_t * visc = NULL;

    if (phi == NULL) {
      pe_info(pe, "viscosity_model arrhenius requires a composition\n");
      pe_fatal(pe, "Please check the fee energy and try again\n");
    }

    /* Parameters */

    rt_double_parameter(rt, "viscosity_arrhenius_eta_plus",  &param.eta_plus);
    rt_double_parameter(rt, "viscosity_arrhenius_eta_minus", &param.eta_minus);
    rt_double_parameter(rt, "viscosity_arrhenius_phistar",   &param.phistar);

    if (param.eta_plus  == 0.0) pe_fatal(pe, "Non-zero eta_plus required\n");
    if (param.eta_minus == 0.0) pe_fatal(pe, "Non-zero eta_minus required\n");
    if (param.phistar   == 0.0) pe_fatal(pe, "Non-zero phistar required\n");

    visc_arrhenius_create(pe, cs, phi, param, &visc);
    ludwig->visc = (visc_t *) visc;

    visc_arrhenius_info(visc);
  }
  else if (key != 0) {
    pe_info(pe, "viscosity_model %s not recognised.\n", description);
    pe_fatal(pe, "Please check and try again.\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  ludwig_colloids_update_low_freq
 *
 *  No rebuild; no lattice operations, but interactions are required.
 *
 *****************************************************************************/

static int ludwig_colloids_update_low_freq(ludwig_t * ludwig) {

  int ncolloid = 0;

  assert(ludwig);

  colloids_info_ntotal(ludwig->collinfo, &ncolloid);
  if (ncolloid == 0) return 0;

  colloids_info_position_update(ludwig->collinfo);
  colloids_info_update_cell_list(ludwig->collinfo);
  colloids_halo_state(ludwig->collinfo);
  colloids_info_update_lists(ludwig->collinfo);

  interact_compute(ludwig->interact, ludwig->collinfo, ludwig->map,
        	     ludwig->psi, ludwig->ewald);

  subgrid_force_from_particles(ludwig->collinfo, ludwig->hydro, ludwig->wall);

  return 0;
}

/*****************************************************************************
 *
 *  ludwig_colloids_update
 *
 *  Driver for update called at start of timestep loop.
 *
 *****************************************************************************/

int ludwig_colloids_update(ludwig_t * ludwig) {

  int ndist;
  int ndevice;
  int ncolloid;
  int iconserve;         /* switch for finite-difference conservation */

  assert(ludwig);

  colloids_info_ntotal(ludwig->collinfo, &ncolloid);
  if (ncolloid == 0) return 0;

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  lb_ndist(ludwig->lb, &ndist);
  iconserve = (ludwig->psi || (ludwig->phi && ndist == 1));

  TIMER_start(TIMER_PARTICLE_HALO);

  colloids_info_position_update(ludwig->collinfo);
  colloids_info_update_cell_list(ludwig->collinfo);
  colloids_halo_state(ludwig->collinfo);
  colloids_info_update_lists(ludwig->collinfo);

  TIMER_stop(TIMER_PARTICLE_HALO);

  /* Removal or replacement of fluid requires a lattice halo update */

  TIMER_start(TIMER_HALO_LATTICE);

  /* __NVCC__ */
  if (ndevice == 0) {
    lb_halo(ludwig->lb);
  }
  else {
    /* Pull data back, then full host halo swap */
    lb_memcpy(ludwig->lb, tdpMemcpyDeviceToHost);
    lb_halo_swap(ludwig->lb, LB_HALO_OPENMP_FULL);
  }

  TIMER_stop(TIMER_HALO_LATTICE);

  if (iconserve && ludwig->phi) field_halo(ludwig->phi);
  if (iconserve && ludwig->psi) psi_halo_rho(ludwig->psi);

  TIMER_start(TIMER_REBUILD);

  build_update_map(ludwig->cs, ludwig->collinfo, ludwig->map);
  build_remove_replace(ludwig->fe, ludwig->collinfo, ludwig->lb, ludwig->phi,
		       ludwig->p, ludwig->q, ludwig->psi, ludwig->map);
  build_update_links(ludwig->cs, ludwig->collinfo, ludwig->wall, ludwig->map,
		     &ludwig->lb->model);

  TIMER_stop(TIMER_REBUILD);

  if (iconserve) {
    colloid_sums_halo(ludwig->collinfo, COLLOID_SUM_CONSERVATION);
    build_conservation(ludwig->collinfo, ludwig->phi, ludwig->psi,
		       &ludwig->lb->model);
  }

  TIMER_start(TIMER_FORCES);

  interact_compute(ludwig->interact, ludwig->collinfo, ludwig->map,
		   ludwig->psi, ludwig->ewald);
  subgrid_force_from_particles(ludwig->collinfo, ludwig->hydro, ludwig->wall);

  TIMER_stop(TIMER_FORCES);


  /* __NVCC__ TODO: remove */

  colloids_memcpy(ludwig->collinfo, tdpMemcpyHostToDevice);
  map_memcpy(ludwig->map, tdpMemcpyHostToDevice);
  lb_memcpy(ludwig->lb, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  io_replace_values
 *
 *  Replaces order parameter values at internal colloid or sites
 *
 *****************************************************************************/

int io_replace_values(field_t * field, map_t * map, int map_id, double value) {

  int ic, jc, kc, index;
  int n, nf;
  int nlocal[3];
  int status;

  assert(field);
  assert(map);

  nf = field->nf;
  cs_nlocal(field->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(field->cs, ic, jc, kc);
        map_status(map, index, &status);

	if (status == map_id) {
	  for (n = 0; n < nf; n++) {
	    field->data[addr_rank1(field->nsites, nf, index, n)] = value;
	  }
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  io_replace_values_kernel
 *
 *  To replace field values for particular map status, e.g., colloid sites.
 *
 *****************************************************************************/

__global__ void io_replace_values_kernel(kernel_3d_t k3d,
					 field_t * field,
					 map_t * map,
					 int status, double value) {
  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {
    int ic = kernel_3d_ic(&k3d, kindex);
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);
    int index = kernel_3d_cs_index(&k3d, ic, jc, kc);

    if (map->status[addr_rank0(map->nsite, index)] == status) {
      for (int n = 0; n < field->nf; n++) {
	field->data[addr_rank1(field->nsites, field->nf, index, n)] = value;
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  io_replace_field_values
 *
 *  Driver to replace specific values in a field.
 *
 *****************************************************************************/

int io_replace_field_values(field_t * field, map_t * map, int status,
			    double value) {

  int nlocal[3] = {0};

  assert(field);
  assert(map);

  cs_nlocal(field->cs, nlocal);

  {
    dim3 nblk = {};
    dim3 ntpb = {};
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    kernel_3d_t k3d = kernel_3d(field->cs, lim);

    kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

    tdpLaunchKernel(io_replace_values_kernel, nblk, ntpb, 0, 0,
		    k3d, field->target, map->target, status, value);

    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());
  }

  return 0;
}

/*****************************************************************************
 *
 *  ludwig_timekeeper_init
 *
 *****************************************************************************/

__host__ int ludwig_timekeeper_init(ludwig_t * ludwig) {

  timekeeper_options_t opts = {0};

  assert(ludwig);

  {
    pe_t * pe = ludwig->pe;
    rt_t * rt = ludwig->rt;

    if (rt_switch(rt, "timer_lap_report")) opts.lap_report = 1;
    rt_int_parameter(rt, "timer_lap_report_freq", &opts.lap_report_freq);

    if (opts.lap_report && opts.lap_report_freq == 0) {
      pe_fatal(pe, "Please specify a timer_lap_report_freq "
	           "(timer_lap_report is on)\n");
    }

    timekeeper_create(pe, &opts, &ludwig->tk);
  }

  return 0;
}

/*****************************************************************************
 *
 *  ludwig_report_statistics
 *
 *  If t = 0, we need to compute any relevant order parameter gradients
 *  for the fist time.
 *  If t > 0 we use the computation coming from the time step loop.
 *
 *****************************************************************************/

int ludwig_report_statistics(ludwig_t * ludwig, int itimestep) {

  assert(ludwig);

  if (itimestep == 0) {
    if (ludwig->phi) {
      field_halo(ludwig->phi);
      field_grad_compute(ludwig->phi_grad);
    }
    if (ludwig->p) {
      field_halo(ludwig->p);
      field_grad_compute(ludwig->p_grad);
    }
    if (ludwig->q) {
      field_halo(ludwig->q);
      field_grad_compute(ludwig->q_grad);
    }
  }

  lb_memcpy(ludwig->lb, tdpMemcpyDeviceToHost);
  stats_distribution_print(ludwig->lb, ludwig->map);

  if (ludwig->phi) {
    field_memcpy(ludwig->phi, tdpMemcpyDeviceToHost);
    field_grad_memcpy(ludwig->phi_grad, tdpMemcpyDeviceToHost);
    if (ludwig->lb->ndist == 2) {
      /* Recompute phi (kernel) and copy back if required */
      phi_lb_to_field(ludwig->phi, ludwig->lb);
      field_memcpy(ludwig->phi, tdpMemcpyDeviceToHost);
      stats_field_info_bbl(ludwig->phi, ludwig->map, ludwig->bbl);
    }
    else {
      if (ludwig->pch) {
	cahn_hilliard_stats(ludwig->pch, ludwig->phi, ludwig->map);
      }
      else {
	field_memcpy(ludwig->phi, tdpMemcpyDeviceToHost);
	stats_field_info(ludwig->phi, ludwig->map);
      }
    }
  }

  if (ludwig->p) {
    /* Get the gradients as well for the free energy below */
    field_memcpy(ludwig->p, tdpMemcpyDeviceToHost);
    field_grad_memcpy(ludwig->p_grad, tdpMemcpyDeviceToHost);
    stats_field_info(ludwig->p, ludwig->map);
  }

  if (ludwig->q) {
    field_memcpy(ludwig->q, tdpMemcpyDeviceToHost);
    field_grad_memcpy(ludwig->q_grad, tdpMemcpyDeviceToHost);
    stats_field_info(ludwig->q, ludwig->map);
    stats_colloid_force_split_output(ludwig->collinfo, itimestep);
  }

  if (ludwig->psi) {
    int ncolloid = 0;
    double psi_zeta = 0.0;
    psi_colloid_rho_set(ludwig->psi, ludwig->collinfo);
    psi_stats_info(ludwig->psi);
    /* Zeta potential for one colloid only to follow psi_stats() */
    /* There should be an explicit option. */
    colloids_info_ntotal(ludwig->collinfo, &ncolloid);
    psi_colloid_zetapotential(ludwig->psi, ludwig->collinfo, &psi_zeta);
    if (ncolloid == 1) pe_info(ludwig->pe, "[psi_zeta] %14.7e\n",  psi_zeta);
  }

  if (ludwig->fe) {
    switch (ludwig->fe->id) {
    case FE_LC:
      fe_lc_stats_info(ludwig->pe, ludwig->cs, ludwig->fe_lc,
		       ludwig->wall, ludwig->map, ludwig->collinfo, itimestep);
      break;
    case FE_TERNARY:
      fe_ternary_stats_info(ludwig->fe_ternary, ludwig->wall,
			    ludwig->map, itimestep);
      break;
    default:
      stats_free_energy_density(ludwig->pe, ludwig->cs, ludwig->wall,
				ludwig->fe, ludwig->map,
				ludwig->collinfo);
    }
  }

  return 0;
}
