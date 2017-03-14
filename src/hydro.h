/*****************************************************************************
 *
 *  hydro.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_HYDRO_H
#define LUDWIG_HYDRO_H

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "io_harness.h"

typedef struct hydro_s hydro_t;

typedef enum {HYDRO_U_HALO_HOST, HYDRO_U_HALO_TARGET} hydro_halo_enum_t;

__host__ int hydro_create(pe_t * pe, cs_t * cs, lees_edw_t * le, 
			  int nhalocomm, hydro_t ** pobj);
__host__ int hydro_free(hydro_t * obj);
__host__ int hydro_init_io_info(hydro_t * obj, int grid[3], int form_in,
				int form_out);
__host__ int hydro_memcpy(hydro_t * ibj, tdpMemcpyKind flag);
__host__ int hydro_io_info(hydro_t * obj, io_info_t ** info);
__host__ int hydro_u_halo(hydro_t * obj);
__host__ int hydro_halo_swap(hydro_t * obj, hydro_halo_enum_t flag);
__host__ int hydro_u_gradient_tensor(hydro_t * obj, int ic, int jc, int kc,
				     double w[3][3]);
__host__ int hydro_lees_edwards(hydro_t * obj);
__host__ int hydro_correct_momentum(hydro_t * obj);
__host__ int hydro_f_zero(hydro_t * obj, const double fzero[3]);
__host__ int hydro_u_zero(hydro_t * obj, const double uzero[3]);

__host__ __device__ int hydro_f_local_set(hydro_t * obj, int index,
					  const double force[3]);
__host__ __device__ int hydro_f_local(hydro_t * obj, int index, double force[3]);
__host__ __device__ int hydro_f_local_add(hydro_t * obj, int index,
					  const double force[3]);
__host__ __device__ int hydro_u_set(hydro_t * obj, int index, const double u[3]);
__host__ __device__ int hydro_u(hydro_t * obj, int index, double u[3]);

#endif
