/****************************************************************************
 *
 *  free_energy_rt.c
 *
 *  Run time initialisation of the free energy, the choice of which
 *  has widespread consequences.
 *
 *  A new free energy will give rise to an addition here.
 *
 *  $Id: free_energy_rt.c,v 1.1.2.2 2009-11-04 18:35:08 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 ****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "symmetric_rt.h"
#include "brazovskii_rt.h"
#include "surfactant_rt.h"
#include "blue_phase_rt.h"
#include "free_energy_rt.h"

/****************************************************************************
 *
 *  free_energy_run_time
 *
 *  The ultimate intension here is that no free energy indicates
 *  single fluid and no order paarameter.
 *
 *  Otherwise, perform the specific initialisation associated with
 *  whatever the user has sellected.
 *
 ****************************************************************************/

void free_energy_run_time(void) {

  int n;
  char description[128];

  n = RUN_get_string_parameter("free_energy", description, 128);

  if (n == 1) {
    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
  }

  if (strcmp(description, "symmetric") == 0) {
    symmetric_run_time();
  }
  else if (strcmp(description, "brazovskii") == 0) {
    brazovskii_run_time();
  }
  else if (strcmp(description, "surfactant") == 0) {
    surfactant_run_time();
  }
  else if (strcmp(description, "lc_blue_phase") == 0) {
    blue_phase_run_time();
  }
  else {
    if (n == 1) {
      /* The user has put something which hasn't been recognised,
       * suggesting a spelling mistake */
      info("free_energy %s not recognised.\n", description);
      fatal("Please check and try again.\n");
    }
  }

  return;
}
