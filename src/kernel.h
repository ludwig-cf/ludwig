/*****************************************************************************
 *
 *  kernel.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_KERNEL_H
#define LUDWIG_KERNEL_H

#include "memory.h"

typedef struct kernel_ctxt_s kernel_ctxt_t;
typedef struct kernel_info_s kernel_info_t;
typedef struct kernel_param_s kernel_param_t;

/* kernel_ctxt_t
 * This is exposed to allow access to the target pointer. */

struct kernel_ctxt_s {
  kernel_param_t * param;
  kernel_ctxt_t * target;
};

/* kernel_info_t
 * is just a convenience to allow the user to pass the
 * relevant information to the context constructor. */

struct kernel_info_s {
  int imin;
  int imax;
  int jmin;
  int jmax;
  int kmin;
  int kmax;
};

__host__ int kernel_ctxt_create(int nsimdvl, kernel_info_t info, kernel_ctxt_t ** p);
__host__ int kernel_ctxt_launch_param(kernel_ctxt_t * obj, dim3 * nblk, dim3 * ntpb);
__host__ int kernel_ctxt_info(kernel_ctxt_t * obj, kernel_info_t * lim);
__host__ int kernel_ctxt_free(kernel_ctxt_t * obj);

__host__ __target__ int kernel_iterations(kernel_ctxt_t * ctxt);
__host__ __target__ int kernel_vector_iterations(kernel_ctxt_t * ctxt);
__host__ __target__ int kernel_baseindex(kernel_ctxt_t * obj, int kindex);
__host__ __target__ int kernel_coords_ic(kernel_ctxt_t * ctxt, int kindex);
__host__ __target__ int kernel_coords_jc(kernel_ctxt_t * ctxt, int kindex);
__host__ __target__ int kernel_coords_kc(kernel_ctxt_t * ctxt, int kindex);
__host__ __target__ int kernel_coords_v(kernel_ctxt_t * ctxt, int kindex,
					int ic[NSIMDVL],
					int jc[NSIMDVL],
					int kc[NSIMDVL]);

__host__ __target__ int kernel_mask(kernel_ctxt_t * ctxt,
				    int ic, int jc, int kc);
__host__ __target__ int kernel_mask_v(kernel_ctxt_t * ctxt,
				      int ic[NSIMDVL], int jc[NSIMDVL],
				      int kc[NSIMDVL], int mask[NSIMDVL]);

__host__ __target__ int kernel_coords_index(kernel_ctxt_t * ctxt,
					    int ic, int jc, int kc);
__host__ __target__ int kernel_coords_index_v(kernel_ctxt_t * ctxt,
					      int ic[NSIMDVL],
					      int jc[NSIMDVL],
					      int kc[NSIMDVL],
					      int index[NSIMDVL]);
/* A "class" method */

__host__ int kernel_launch_param(int iterations, dim3 * nblk, dim3 * ntpb);

#endif

