/*****************************************************************************
 *
 *  lb_d2q9.h
 *
 *  D2Q9 definition. See lb_d2q9.c for details.
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

#ifndef LUDWIG_LB_D2Q9_H
#define LUDWIG_LB_D2Q9_H

#include "lb_model.h"

/* Velocity set and weights. */

enum {NVEL_D2Q9 = 9};

#define LB_CV_D2Q9(cv) const int8_t cv[NVEL_D2Q9][3] = {      \
    { 0,  0,  0},                                             \
    { 1,  1,  0}, { 1,  0,  0}, { 1, -1,  0}, { 0,  1,  0},   \
    { 0, -1,  0}, {-1,  1,  0}, {-1,  0,  0}, {-1, -1,  0}};

#define LB_WEIGHTS_D2Q9(wv) const double wv[NVEL_D2Q9] = {    \
   16.0/36.0,                                                 \
    1.0/36.0, 4.0/36.0, 1.0/36.0, 4.0/36.0,                   \
    4.0/36.0, 1.0/36.0, 4.0/36.0, 1.0/36.0};

int lb_d2q9_create(lb_model_t * model);

#endif
