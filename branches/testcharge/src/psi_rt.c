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
#include "psi_s.h"
#include "psi_rt.h"
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
  int io_grid[3] = {1,1,1};

  char filename[FILENAME_MAX];

  /* Local reference */
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

  n = RUN_get_int_parameter_vector("default_io_grid", io_grid);
  psi_init_io_info(psi,io_grid);

  n = RUN_get_string_parameter("psi_format", filename, FILENAME_MAX);
  if (strcmp(filename, "ASCII") == 0) {
    io_info_set_format_ascii(psi->info);
    info("Setting psi I/O format to ASCII\n");
  }


  //fatal("Not ready yet\n");

  return 0;
}
