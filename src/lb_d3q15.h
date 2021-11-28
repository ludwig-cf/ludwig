/*****************************************************************************
 *
 *  lb_d3q15.h
 *
 *  D3Q15 definition. See lb_d3q15.c for details.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_D3Q15_H
#define LUDWIG_LB_D3Q15_H

#include "lb_model.h"

/* Velocity set and weights.*/

enum {NVEL_D3Q15 = 15};

#define LB_CV_D3Q15(cv) const int8_t cv[NVEL_D3Q15][3] = {      \
    { 0,  0,  0},                                               \
    { 1,  1,  1}, { 1,  1, -1}, { 1,  0,  0},                   \
    { 1, -1,  1}, { 1, -1, -1}, { 0,  1,  0},                   \
    { 0,  0,  1}, { 0,  0, -1}, { 0, -1,  0},                   \
    {-1,  1,  1}, {-1,  1, -1}, {-1,  0,  0},                   \
    {-1, -1,  1}, {-1, -1, -1}};

#define LB_WEIGHTS_D3Q15(wv) const double wv[NVEL_D3Q15] = {   16.0/72.0, \
    1.0/72.0, 1.0/72.0, 8.0/72.0, 1.0/72.0, 1.0/72.0, 8.0/72.0, 8.0/72.0, \
    8.0/72.0, 8.0/72.0, 1.0/72.0, 1.0/72.0, 8.0/72.0, 1.0/72.0, 1.0/72.0 };

int lb_d3q15_create(lb_model_t * model);

#endif
