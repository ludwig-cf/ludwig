/*****************************************************************************
 *
 *  tests.c
 *
 *  This runs the tests.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "tests.h"

int tests_create(void);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);

  tests_create();

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  tests_create
 *
 *****************************************************************************/

int tests_create() {

  test_pe_suite();
  test_coords_suite();

  test_kernel_suite();
  test_field_suite();

  test_angle_cosine_suite();
  test_assumptions_suite();
  test_be_suite();
  test_bond_fene_suite();
  test_bonds_suite();
  test_build_suite();
  test_colloid_suite();
  test_colloid_sums_suite();
  test_colloids_info_suite();
  test_colloids_halo_suite();
  test_ewald_suite();
  test_fe_electro_suite();
  test_fe_electro_symm_suite();
  test_field_suite();
  test_field_grad_suite();           
  test_halo_suite();
  test_hydro_suite();
  test_io_suite();
  test_le_suite();
  test_lubrication_suite();
  test_map_suite();
  test_model_suite();
  test_noise_suite();
  test_pair_lj_cut_suite();
  test_pair_ss_cut_suite();
  test_pair_yukawa_suite();
  test_polar_active_suite();
  test_psi_suite();
  test_lb_prop_suite();
  test_random_suite();
  test_rt_suite();
  test_timer_suite();
  test_util_suite();

  /* Failing... pending investigation */

  /* test_nernst_planck_suite(); */
  /* test_psi_sor_suite();*/
  /* test_phi_ch_suite(); replace by advection without CH */
  /* test_bp_suite(); Needs attention to electric field */


  return 0;
}

/*****************************************************************************
 *
 *  test_assert
 *
 *  Asimple assertion to control what happens in parallel.
 *
 *****************************************************************************/

void test_assert(const int lvalue) {

  int rank;

  if (lvalue) {
    /* ok */
  }
  else {
    /* Who has failed? */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[%d] ***************** Failed test assertion\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  return;
}
