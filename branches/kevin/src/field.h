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

#define NVECTOR 3    /* Storage requirement for vector (per site) */
#define NQAB 5       /* Storage requirement for symmetric, traceless tensor */

#include "io_harness.h"
#include "targetDP.h"

typedef struct field_s field_t;

HOST int field_create(int nf, const char * name, field_t ** pobj);
HOST void field_free(field_t * obj);

HOST int field_init(field_t * obj, int nhcomm);
HOST int field_nf(field_t * obj, int * nop);
HOST int field_init_io_info(field_t * obj, int grid[3], int form_in, int form_out);
HOST int field_io_info(field_t * obj, io_info_t ** info);
HOST int field_halo(field_t * obj);
HOST int field_leesedwards(field_t * obj);

HOST int field_scalar(field_t * obj, int index, double * phi);
HOST int field_scalar_set(field_t * obj, int index, double phi);
HOST int field_vector(field_t * obj, int index, double p[3]);
HOST int field_vector_set(field_t * obj, int index, const double p[3]);
HOST int field_tensor(field_t * obj, int index, double q[3][3]);
HOST int field_tensor_set(field_t * obj, int index, double q[3][3]);

HOST int field_scalar_array(field_t * obj, int index, double * array);
HOST int field_scalar_array_set(field_t * obj, int index, const double * array);

#endif
