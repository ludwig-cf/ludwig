/*****************************************************************************
 *
 *  tests.c
 *
 *  This runs the tests.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "tests.h"

__host__ int tests_create(int argc, char ** argv);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

__host__ int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);

  tests_create(argc, argv);

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  tests_create
 *
 *****************************************************************************/

__host__ int tests_create(int argc, char ** argv) {

  test_pe_suite();
  test_coords_suite();
  test_cs_limits_suite();

  test_kernel_suite();
  test_kernel_3d_suite();

  test_gradient_d3q27_suite();
  test_angle_cosine_suite();
  test_assumptions_suite();
  test_be_suite();
  test_bond_fene_suite();
  test_bonds_suite();
  test_bp_suite();
  test_build_suite();
  test_ch_suite();
  test_colloid_suite();
  test_colloid_sums_suite();
  test_colloids_info_suite();
  test_colloids_halo_suite();
  test_ewald_suite();
  test_fe_null_suite();
  test_fe_electro_suite();
  test_fe_electro_symm_suite();
  test_fe_lc_droplet_suite();
  test_fe_force_method_suite();
  test_fe_force_method_rt_suite();
  test_field_suite();
  test_field_grad_suite();
  test_halo_suite();
  test_hydro_options_suite();
  test_hydro_suite();
  test_interaction_suite();
  test_io_aggregator_suite();
  test_io_element_suite();
  test_io_options_suite();
  test_io_options_rt_suite();
  test_io_info_args_suite();
  test_io_info_args_rt_suite();
  test_io_subfile_suite();
  test_io_metadata_suite();
  test_io_impl_mpio_suite();
  test_io_suite();
  test_lb_d2q9_suite();
  test_lb_d3q15_suite();
  test_lb_d3q19_suite();
  test_lb_d3q27_suite();
  test_lb_model_suite();
  test_lb_bc_inflow_opts_suite();
  test_lb_bc_inflow_rhou_suite();
  test_lb_bc_outflow_opts_suite();
  test_lb_bc_outflow_rhou_suite();
  test_lc_anchoring_suite();
  test_le_suite();
  test_lubrication_suite();
  test_map_suite();
  test_map_init_suite();
  test_model_suite();
  test_noise_suite();
  test_pair_lj_cut_suite();
  test_pair_ss_cut_suite();
  test_pair_ss_cut_ij_suite();
  test_pair_yukawa_suite();
  test_phi_bc_inflow_opts_suite();
  test_phi_bc_inflow_fixed_suite();
  test_phi_bc_outflow_opts_suite();
  test_phi_bc_outflow_free_suite();
  test_phi_ch_suite();
  test_polar_active_suite();

  test_psi_solver_options_suite(argc, argv);
  test_psi_options_suite();
  test_psi_suite();
  test_psi_solver_petsc_suite();
  test_psi_sor_suite();
  test_nernst_planck_suite();
  test_lb_prop_suite();
  test_random_suite();
  test_rt_suite();
  test_stencil_d3q7_suite();
  test_stencil_d3q19_suite();
  test_stencil_d3q27_suite();
  test_stencils_suite();
  test_timer_suite();
  test_util_suite();
  test_util_bits_suite();
  test_util_ellipsoid_suite();
  test_util_fopen_suite();
  test_util_io_suite();
  test_util_json_suite();
  test_util_sum_suite();
  test_util_vector_suite();
  test_visc_arrhenius_suite();
  test_wall_suite();
  test_wall_ss_cut_suite();

  test_fe_surfactant1_suite();
  test_fe_symmetric_suite();
  test_fe_ternary_suite();

  return 0;
}

/*****************************************************************************
 *
 *  test_assert
 *
 *  Asimple assertion to control what happens in parallel.
 *
 *****************************************************************************/

__host__ __device__ void test_assert_info(const int lvalue, int line,
					  const char * file) {

  if (lvalue) {
    /* ok */
  }
  else {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
    /* No rank available */
    printf("Line %d file %s Failed test assertion\n", line, file);
    assert(0);
#else
    int rank;

    /* Who has failed? */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[%d] Line %d file %s Failed test assertion\n", rank, line, file);
    MPI_Abort(MPI_COMM_WORLD, 0);
#endif
  }

  return;
}
