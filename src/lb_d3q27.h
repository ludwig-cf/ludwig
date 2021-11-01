/*****************************************************************************
 *
 *  lb_d3q27.h
 *
 *  D3Q27 model definition. See lb_d3q27.c for details.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Comuting Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_D3Q27_H
#define LUDWIG_LB_D3Q27_H

#include "lb_model.h"

/* Velocity set, weights. */

enum {NVEL_D3Q27 = 27};

#define LB_CV_D3Q27(cv) const int8_t cv[NVEL_D3Q27][3] = {     { 0, 0, 0}, \
   {-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1}, {-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1}, \
   {-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1}, { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1}, \
   { 0, 0,-1},                                                 { 0, 0, 1}, \
   { 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1}, { 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1}, \
   { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1}, { 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}};

/* Weights: |0| = 64 |1| = 16 |2| = 1 |3| = 4.  All / 216 */

#define LB_WEIGHTS_D3Q27(wv) const double wv[NVEL_D3Q27] = {   64.0/216.0, \
      4.0/216.0,  1.0/216.0, 4.0/216.0, 1.0/216.0, 16.0/216.0,  1.0/216.0, \
      4.0/216.0,  1.0/216.0, 4.0/216.0, 1.0/216.0, 16.0/216.0,  1.0/216.0, \
     16.0/216.0,                                               16.0/216.0, \
      1.0/216.0, 16.0/216.0, 1.0/216.0, 4.0/216.0,  1.0/216.0,  4.0/216.0, \
      1.0/216.0, 16.0/216.0, 1.0/216.0, 4.0/216.0,  1.0/216.0,  4.0/216.0};

/* Normalisers TBC */

int lb_d3q27_create(lb_model_t * model);

#endif
