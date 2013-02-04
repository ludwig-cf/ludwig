/*****************************************************************************
 * 
 * field_datamgmt_internal_gpu.h
 * 
 * Field data management utilities for GPU adaptation of Ludwig
 * Alan Gray
 *
 *****************************************************************************/

#ifndef FIELDDATA_INTERNAL_GPU_H
#define FIELDDATA_INTERNAL_GPU_H

#include "common_gpu.h"
#include "colloids.h"
/* declarations for required external (host) routines */
extern "C" double phi_get_phi_site(const int);
extern "C" double phi_op_get_phi_site(const int index, const int nop);
extern "C" void phi_set_phi_site(const int, const double);
extern "C" void phi_op_set_phi_site(const int, const int, const double);
extern "C" void   phi_gradients_grad_n(const int index, const int iop, double grad[3]);
extern "C" void   phi_gradients_set_grad_n(const int index, const int iop, double grad[3]);
extern "C" double phi_gradients_delsq_n(const int index, const int iop);
extern "C" void phi_gradients_set_delsq_n(const int index, const int iop, const double delsq);
extern "C" void TIMER_start(const int);
extern "C" void TIMER_stop(const int);
extern "C" void halo_gpu(int nfields1, int nfields2, int packfield1, double * data_d);
extern "C" void put_field_partial_on_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *));

extern "C" void get_field_partial_from_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *));
extern "C" void hydrodynamics_get_force_local(const int, double *);
extern "C" void hydrodynamics_set_force_local(const int, double *);
extern "C" void hydrodynamics_get_velocity(const int, double *);
extern "C" void hydrodynamics_set_velocity(const int, double *);
extern "C" colloid_t * colloid_at_site_index(int index);


/* forward declarations of host routines internal to this module */
static void calculate_field_data_sizes(void);
static void allocate_field_memory_on_gpu(void);
static void free_field_memory_on_gpu(void);


#endif

