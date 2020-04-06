/*****************************************************************************
 *
 *  tests.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2019 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_UNIT_TESTS_H
#define LUDWIG_UNIT_TESTS_H

#include "target.h"

__host__ __device__ void test_assert_info(const int lvalue, int line,
					  const char * file);

#define test_assert(x) test_assert_info((x), __LINE__, __FILE__)

#define TEST_FLOAT_TOLERANCE  1.0e-07
#define TEST_DOUBLE_TOLERANCE 1.0e-14

/* List of test drivers (see relevant file.c) */

int test_angle_cosine_suite(void);
int test_assumptions_suite(void);
int test_be_suite(void);
int test_bp_suite(void);
int test_bond_fene_suite(void);
int test_bonds_suite(void);
int test_build_suite(void);
int test_colloid_sums_suite(void);
int test_colloid_suite(void);
int test_colloids_info_suite(void);
int test_colloids_halo_suite(void);
int test_coords_suite(void);
int test_ewald_suite(void);
int test_fe_electro_suite(void);
int test_fe_electro_symm_suite(void);
int test_field_suite(void);
int test_field_grad_suite(void);
int test_halo_suite(void);
int test_hydro_suite(void);
int test_io_suite(void);
int test_le_suite(void);
int test_kernel_suite(void);
int test_lubrication_suite(void);
int test_map_suite(void);
int test_model_suite(void);
int test_nernst_planck_suite(void);
int test_noise_suite(void);
int test_pair_lj_cut_suite(void);
int test_pair_ss_cut_suite(void);
int test_pair_yukawa_suite(void);
int test_pe_suite(void);
int test_phi_ch_suite(void);
int test_polar_active_suite(void);
int test_lb_prop_suite(void);
int test_psi_suite(void);
int test_psi_sor_suite(void);
int test_random_suite(void);
int test_rt_suite(void);
int test_timer_suite(void);
int test_util_suite(void);

#endif
