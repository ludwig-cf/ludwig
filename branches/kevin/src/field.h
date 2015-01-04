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
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef FIELD_H
#define FIELD_H

#define NPAIR 2      /* Storage requirement for pair field */
#define NVECTOR 3    /* Storage requirement for vector (per site) */
#define NQAB 5       /* Storage requirement for symmetric, traceless tensor */

#include "coords.h"
#include "leesedwards.h"
#include "io_harness.h"

typedef struct field_s field_t;

__host__ int field_create(coords_t * cs, int nf, const char * name,
			  field_t ** pobj);
__host__ int field_free(field_t * obj);

__host__ int field_init(field_t * obj, int nhcomm, le_t * le);
__host__ int field_nf(field_t * obj, int * nop);
__host__ int field_init_io_info(field_t * obj, int grid[3], int form_in,
				int form_out);
__host__ int field_io_info(field_t * obj, io_info_t ** info);
__host__ int field_halo(field_t * obj);
__host__ int field_leesedwards(field_t * obj);

__host__ int field_scalar(field_t * obj, int index, double * phi);
__host__ int field_scalar_set(field_t * obj, int index, double phi);
__host__ int field_vector(field_t * obj, int index, double p[3]);
__host__ int field_vector_set(field_t * obj, int index, const double p[3]);
__host__ int field_tensor(field_t * obj, int index, double q[3][3]);
__host__ int field_tensor_set(field_t * obj, int index, double q[3][3]);

__host__ int field_scalar_array(field_t * obj, int index, double * array);
__host__ int field_scalar_array_set(field_t * obj, int index,
				    const double * array);

#endif
