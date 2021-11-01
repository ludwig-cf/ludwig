/*****************************************************************************
 *
 *  lb_model.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_MODEL_H
#define LUDWIG_LB_MODEL_H

/* This file is scheduled to replace lb_model_s.h and model.h */
/* and will only describe general fixed model parameters. */

#include <stdint.h>

#include "cartesian.h"  /* Not used here explicitly but required elsewhere */

typedef struct lb_model_s lb_model_t;

struct lb_model_s {
  int8_t ndim;          /* Usually 2 or 3 */
  int8_t nvel;          /* 0 <= p < NVEL */
  int8_t (* cv)[3];     /* cv[p][3] always 3d */
  double * wv;          /* weights wv[p] */
  double * na;          /* normalisers na[p] */
  double ** ma;         /* Matrix M^a eigenvctors of collision matrix */
  double cs2;           /* (speed of sound)^2 */
};

int lb_model_create(int nvel, lb_model_t * model);
int lb_model_free(lb_model_t * model);

#endif
