/*****************************************************************************
 * 
 * field_datamgmt_gpu.h
 * 
 * Field data management utilities for GPU adaptation of Ludwig
 * Alan Gray
 *
 *****************************************************************************/

#ifndef FIELDDATA_GPU_H
#define FIELDDATA_GPU_H

#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif

/* expose routines in this module to outside routines */
CFUNC void put_all_fields_on_gpu();
CFUNC void get_all_fields_from_gpu();
CFUNC void put_f_on_gpu();
CFUNC void get_f_from_gpu();
CFUNC void put_phi_on_gpu();
CFUNC void put_grad_phi_on_gpu();
CFUNC void put_delsq_phi_on_gpu();
CFUNC void get_phi_from_gpu();
CFUNC void get_grad_phi_from_gpu();
CFUNC void get_delsq_phi_from_gpu();
CFUNC void put_force_on_gpu();
CFUNC void put_velocity_on_gpu();
CFUNC void get_force_from_gpu();
CFUNC void zero_force_on_gpu();
CFUNC void get_velocity_from_gpu();
CFUNC void phi_halo_gpu(void);
CFUNC void velocity_halo_gpu(void);
CFUNC void distribution_halo_gpu(void);
CFUNC void expand_phi_on_gpu();
CFUNC void expand_grad_phi_on_gpu();
CFUNC void put_velocity_partial_on_gpu(int include_neighbours);
CFUNC void get_velocity_partial_from_gpu(int include_neighbours);
CFUNC void put_phi_partial_on_gpu(int include_neighbours);
CFUNC void get_phi_partial_from_gpu(int include_neighbours);
CFUNC void put_f_partial_on_gpu(int include_neighbours);
CFUNC void get_f_partial_from_gpu(int include_neighbours);
CFUNC void switch_f_and_ftmp_on_gpu(void);
CFUNC void copy_f_to_ftmp_on_gpu(void);
CFUNC void update_colloid_force_from_gpu();
CFUNC void init_field_gpu();
CFUNC void finalise_field_gpu();
CFUNC double sum_f_from_gpu();
CFUNC double sum_phi_from_gpu();
CFUNC double sum_grad_phi_from_gpu();
CFUNC double sum_delsq_phi_from_gpu();
CFUNC double sum_force_from_gpu();
CFUNC double sum_velocity_from_gpu();

#endif

