/*****************************************************************************
 *
 *  lc_anchoring.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LC_ANCHORING_H
#define LUDWIG_LC_ANCHORING_H

/* Surface anchoring types and associated structure */

typedef enum lc_anchoring_enum {
  LC_ANCHORING_NONE = 0,
  LC_ANCHORING_PLANAR,
  LC_ANCHORING_NORMAL, 
  LC_ANCHORING_FIXED,
  LC_ANCHORING_INVALID /* Last entry */
} lc_anchoring_enum_t;

typedef struct lc_anchoring_param_s lc_anchoring_param_t;

struct lc_anchoring_param_s {
  lc_anchoring_enum_t type;      /* normal, planar, etc */
  double w1;                     /* Free energy constant w1 */
  double w2;                     /* Free energy constant w2 (planar only) */
  double nfix[3];                /* Preferred director (fixed type only) */
};

const char * lc_anchoring_type_from_enum(lc_anchoring_enum_t type);
lc_anchoring_enum_t lc_anchoring_type_from_string(const char * str);

/* Matrices for gradient terms in the boundary condition */

typedef struct lc_anchoring_matrices_s lc_anchoring_matrices_t;

struct lc_anchoring_matrices_s {
  double a6inv[3][6];              /* Single "unknown" (faces) */
  double a12inv[3][12][12];        /* Two unknowns (edges) */
  double a18inv[18][18];           /* Three unknowns (corners) */
};

int lc_anchoring_matrix1(double kappa0, double kappa1, double a6[3][6]);
int lc_anchoring_matrix2(double kappa0, double kappa1, double a12[3][12][12]);
int lc_anchoring_matrix3(double kappa0, double kappa1, double a18[18][18]);
int lc_anchoring_matrices(double kappa0, double kappa1,
			  lc_anchoring_matrices_t * matrices);

/* Inline */

#include "lc_anchoring_impl.h"

#endif
