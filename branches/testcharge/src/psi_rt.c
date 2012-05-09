/*****************************************************************************
 *
 *  psi_rt.c
 *
 *  Run time initialisation of electrokinetics stuff.
 *
 *  At the moment the number of species is set to 2 automatically
 *  if the electrokinetics is switched on.
 *
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "psi.h"
#include "psi_rt.h"
#include "psi_init.h"
#include "io_harness.h"

static int psi_do_init(void);

/*****************************************************************************
 *
 *  psi_init_rt
 *
 *****************************************************************************/

int psi_init_rt(void) {

  int eswitch =  0;
  char str[BUFSIZ];

  if (RUN_get_string_parameter("electrokinetics", str, BUFSIZ)) {
    if (strcmp(str, "on") == 0) eswitch = 1;
    if (strcmp(str, "yes") == 0) eswitch = 1;
    if (strcmp(str, "1") == 0) eswitch = 1;
  }

  info("\n");
  info("Electrokinetics using Nernst Planck\n");
  info("-----------------------------------\n");
  info("Electrokinetics: %s\n", (eswitch) ? "on" : "off");

  if (eswitch) psi_do_init();

  return 0;
}

/*****************************************************************************
 *
 *  psi_do_init
 *
 *****************************************************************************/

static int psi_do_init(void) {

  int n;
  int nk = 2;                 /* Number of charge densities always 2 for now */

  int valency[2] = {+1, -1};  /* Valencies (should be +/-!)*/
  double diffusivity[2] = {0.01, 0.01};

  double eunit = 1.0;         /* Unit charge */
  double temperature, beta;   /* Temperature (set by fluctuations) */
  double epsilon = 0.0;       /* Permeativity */
  double lb;                  /* Bjerrum length; derived, not input. */
  double tolerance;           /* Numerical tolerance for SOR. */
  double rho_el;              /* Charge density */
  double delta_el;            /* Relative difference in charge densities */
  double sigma;               /* Surface charge density */

  int io_grid[3] = {1,1,1};
  int io_format_in = IO_FORMAT_DEFAULT;
  int io_format_out = IO_FORMAT_DEFAULT;
  char value[BUFSIZ] = "BINARY";

  /* Local reference; initialise psi_ via psi_init(). */
  psi_t * psi = NULL; 

  psi_init(2, &psi);
  assert(psi);

  n = RUN_get_int_parameter("electrokinetics_z0", valency);
  n = RUN_get_int_parameter("electrokinetics_z1", valency + 1);
  n = RUN_get_double_parameter("electrokinetics_d0", diffusivity);
  n = RUN_get_double_parameter("electrokinetics_d1", diffusivity + 1);

  for (n = 0; n < nk; n++) {
    psi_valency_set(psi, n, valency[n]);
    psi_diffusivity_set(psi, n, diffusivity[n]);
  }

  n = RUN_get_double_parameter("electrokinetics_eunit", &eunit);
  n = RUN_get_double_parameter("electrokinetics_epsilon", &epsilon);

  psi_unit_charge_set(psi, eunit);
  psi_epsilon_set(psi, epsilon);

  n = RUN_get_double_parameter("temperature", &temperature);

  if (n == 0 || temperature <= 0.0) {
    fatal("Please set a temperature to use electrokinetics\n");
  }

  beta = 1.0/temperature;

  psi_beta_set(psi, beta);
  psi_bjerrum_length(psi, &lb);

  info("Electrokinetic species:    %2d\n", nk);
  info("Boltzmann factor:          %14.7e (T = %14.7e)\n", beta, temperature);
  info("Unit charge:               %14.7e\n", eunit);
  info("Reference permeativity:    %14.7e\n", epsilon);
  info("Bjerrum length:            %14.7e\n", lb);

  for (n = 0; n < nk; n++) {
    info("Valency species %d:         %2d\n", n, valency[n]);
    info("Diffusivity species %d:     %14.7e\n", n, diffusivity[n]);
  }

  /* Tolerances. Yet to be offered from input */

  psi_reltol(psi, &tolerance);
  info("Relative tolerance (SOR):  %14.7e\n", tolerance);
  psi_abstol(psi, &tolerance);
  info("Absolute Tolerance (SOR):  %14.7e\n", tolerance);

  /* I/O */

  n = RUN_get_int_parameter_vector("default_io_grid", io_grid);
  n = RUN_get_string_parameter("psi_format", value, BUFSIZ);

  if (strcmp(value, "ASCII") == 0) {
    io_format_in = IO_FORMAT_ASCII;
    io_format_out = IO_FORMAT_ASCII;
  }

  info("I/O decomposition:          %d %d %d\n", io_grid[0], io_grid[1],
       io_grid[2]);
  info("I/O format:                 %s\n", value);

  psi_init_io_info(psi, io_grid, io_format_in, io_format_out);

  /* Initial charge densities */

  n = RUN_get_string_parameter("electrokinetics_init", value, BUFSIZ);

  if (strcmp(value, "gouy_chapman") == 0) {
    info("Initial conditions:         %s\n", "Gouy Chapman");

    n = RUN_get_double_parameter("electrokinetics_init_rho_el", &rho_el);
    if (n == 0) fatal("... please set electrokinetics_rho_el\n");
    info("Initial condition rho_el: %14.7e\n", rho_el);

    n = RUN_get_double_parameter("electrokinetics_init_sigma", &sigma);
    if (n == 0) fatal("... please set electrokinetics_sigma\n");
    info("Initial condition sigma: %14.7e\n", sigma);

    psi_init_gouy_chapman_set(psi, rho_el, sigma);
  }

  if (strcmp(value, "liquid_junction") == 0) {
    info("Initial conditions:         %s\n", "Liquid junction");

    n = RUN_get_double_parameter("electrokinetics_init_rho_el", &rho_el);
    if (n == 0) fatal("... please set electrokinetics_rho_el\n");
    info("Initial condition rho_el: %14.7e\n", rho_el);

    n = RUN_get_double_parameter("electrokinetics_init_delta_el", &delta_el);
    if (n == 0) fatal("... please set electrokinetics_delta_el\n");
    info("Initial condition delta_el: %14.7e\n", delta_el);

    psi_init_liquid_junction_set(psi, rho_el, delta_el);
  }

  return 0;
}
