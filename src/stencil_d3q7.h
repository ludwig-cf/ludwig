/*****************************************************************************
 *
 *  stencil_d3q7.h
 *
 *  A 7-point stencil inn three dimensions.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_STENCIL_D3Q7_H
#define LUDWIG_STENCIL_D3Q7_H

#include <stdint.h>
#include "stencil.h"

enum {NVEL_D3Q7 = 7};

#define LB_CV_D3Q7(cv) const int8_t cv[NVEL_D3Q7][3] = { \
{0,0,0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}, {-1, 0, 0}};

/* The weights are |0| = 2 and |1| = 1 (all over 8). */
#define LB_WEIGHTS_D3Q7(wv) const double wv[NVEL_D3Q7] = { \
2.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0 };

int stencil_d3q7_create(stencil_t ** s);

#endif
