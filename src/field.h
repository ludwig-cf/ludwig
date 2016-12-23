/*****************************************************************************
 *
 *  field.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_H
#define LUDWIG_FIELD_H

#define NVECTOR 3    /* Storage requirement for vector (per site) */
#define NQAB 5       /* Storage requirement for symmetric, traceless tensor */

#include "pe.h"
#include "coords.h"
#include "io_harness.h"
#include "leesedwards.h"

typedef struct field_s field_t;

typedef enum {FIELD_HALO_HOST = 0, FIELD_HALO_TARGET} field_halo_enum_t;

__host__ int field_create(pe_t * pe, cs_t * cs, int nf, const char * name,
			  field_t ** pobj);
__host__ int field_free(field_t * obj);

__host__ int field_memcpy(field_t * obj, int flag);
__host__ int field_init(field_t * obj, int nhcomm, lees_edw_t * le);
__host__ int field_init_io_info(field_t * obj, int grid[3], int form_in,
				int form_out);
__host__ int field_io_info(field_t * obj, io_info_t ** info);
__host__ int field_halo(field_t * obj);
__host__ int field_halo_swap(field_t * obj, field_halo_enum_t flag);
__host__ int field_leesedwards(field_t * obj);

__host__ __device__ int field_nf(field_t * obj, int * nop);
__host__ __device__ int field_scalar(field_t * obj, int index, double * phi);
__host__ __device__ int field_scalar_set(field_t * obj, int index, double phi);
__host__ __device__ int field_vector(field_t * obj, int index, double p[3]);
__host__ __device__ int field_vector_set(field_t * obj, int index,
					 const double p[3]);
__host__ __device__ int field_tensor(field_t * obj, int index, double q[3][3]);
__host__ __device__ int field_tensor_set(field_t * obj, int index,
					 double q[3][3]);
__host__ __device__ int field_scalar_array(field_t * obj, int index,
					   double * array);
__host__ __device__ int field_scalar_array_set(field_t * obj, int index,
					       const double * array);


#endif
