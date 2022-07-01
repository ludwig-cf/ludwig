/*****************************************************************************
 *
 *  lc_anchoring.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LC_ANCHORING_H
#define LUDWIG_LC_ANCHORING_H

/* Surface anchoring types */
typedef enum lc_anchoring_enum {
  LC_ANCHORING_INVALID = 0,
  LC_ANCHORING_PLANAR,
  LC_ANCHORING_NORMAL, 
  LC_ANCHORING_FIXED,
  LC_ANCHORING_NTYPES /* Last entry */
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

#endif
