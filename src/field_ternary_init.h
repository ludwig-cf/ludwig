/*****************************************************************************
 *
 *  field_ternary_init.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_TERNARY_INIT_H
#define LUDWIG_FIELD_TERNARY_INIT_H

#include "field.h"

/* Containers to hold various initialisation parameters */

typedef struct fti_block_s fti_block_t;
typedef struct fti_drop_s  fti_drop_t;

struct fti_block_s {
  double xf1, xf2, xf3;     /* Fractions of system length 0 < f < 1 */
  double yf1, yf2, yf3;     /* Ditto */
  double zf1, zf2, zf3;     /* Ditto */
};

struct fti_drop_s {
  double r0[3];             /* Centre of drop (x0,y0,z0) */
  double r;                 /* Radius of drop */
};

int field_ternary_init_X(field_t * phi);
int field_ternary_init_2d_double_emulsion(field_t * phi,
					  const fti_block_t * block);
int field_ternary_init_2d_tee(field_t * phi, const fti_block_t *);
int field_ternary_init_2d_lens(field_t * phi, const fti_drop_t * drop);
int field_ternary_init_2d_double_drop(field_t * phi, const fti_drop_t * drop1,
				      const fti_drop_t * drop2);

#endif
