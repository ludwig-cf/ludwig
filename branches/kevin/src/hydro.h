/*****************************************************************************
 *
 *  hydro.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef HYDRO_H
#define HYDRO_H

#include "coords.h"
#include "leesedwards.h"
#include "io_harness.h"

typedef struct hydro_s hydro_t;

__host__ int hydro_create(coords_t * cs, int nhalocomm, le_t * le,
			  hydro_t ** pobj);
__host__ int hydro_free(hydro_t * obj);
__host__ int hydro_init_io_info(hydro_t * obj, int grid[3], int form_in,
				int form_out);
__host__ int hydro_io_info(hydro_t * obj, io_info_t ** info);

__host__ int hydro_u_halo(hydro_t * obj);
__host__ int hydro_f_local_set(hydro_t * obj, int index, const double force[3]);
__host__ int hydro_f_local(hydro_t * obj, int index, double force[3]);
__host__ int hydro_f_local_add(hydro_t * obj, int index, const double force[3]);
__host__ int hydro_u_set(hydro_t * obj, int index, const double u[3]);
__host__ int hydro_u(hydro_t * obj, int index, double u[3]);
__host__ int hydro_f_zero(hydro_t * obj, const double fzero[3]);
__host__ int hydro_u_zero(hydro_t * obj, const double uzero[3]);
__host__ int hydro_u_gradient_tensor(hydro_t * obj, int ic, int jc, int kc,
				     double w[3][3]);
__host__ int hydro_lees_edwards(hydro_t * obj);

#endif
